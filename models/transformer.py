from models.model_utils import get_dict_info_batch

import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
import math
import pdb
import copy
import random

def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        output_custom = torch.log(x_exp / x_exp_sum)
    else:
        output_custom = x_exp / x_exp_sum
    return output_custom

class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        # input_dim = 3
        x_id_size = parameters.max_xid + 1
        y_id_size = parameters.max_yid + 1
        self.emb_id_x = nn.Embedding(x_id_size, self.hid_dim)
        self.emb_id_y = nn.Embedding(y_id_size, self.hid_dim)
        self.in_proj = nn.Linear(self.hid_dim * 2 + 1, self.hid_dim)

    def forward(self, src):
        x = self.emb_id_x(src[:,:,0].long())
        y = self.emb_id_y(src[:,:,1].long())
        input = torch.cat((x,y,src[:,:,2:]), dim=2)
        outputs = self.in_proj(input)
        return outputs
# BART Encoder-Decoder模型

class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()

        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag

        self.emb_id = nn.Embedding(self.id_size, self.id_emb_dim)
        self.device = parameters.device
        
        # rnn_input_dim = self.id_emb_dim + 1
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim
        
        type_input_dim = self.id_emb_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
                          nn.Linear(type_input_dim, self.hid_dim),
                          nn.ReLU()
                          )
            
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)
        
        
    def forward(self, outputs, mask = None):
        # pre_rid
        id_logits = self.fc_id_out(outputs)
        if mask is not None:
            id_logits = id_logits + mask

        prediction_id = F.log_softmax(id_logits, dim=2)
        # pre_rate
        max_id = prediction_id.argmax(dim=2).long()
        id_emb = self.dropout(self.emb_id(max_id))
        rate_input = torch.cat((id_emb, outputs),dim=2)
        rate_input = self.tandem_fc(rate_input)
        prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        return prediction_id, prediction_rate


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model//2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 1::2] = torch.cos(position * div_term.unsqueeze(0))
        if d_model % 2 != 0:
            pe[:, 0:-1:2] = torch.sin(position * div_term.unsqueeze(0))
            last_term = torch.tensor([-math.log(10000.0) * (d_model - 1) / (2 * d_model)]).float()
            # exp_last_term = torch.exp(last_term)
            # exp_last_term = exp_last_term.expand_as(position)
            pe[:, -1:] = torch.sin(position * torch.exp(last_term))
        else:
            pe[:, 0::2] = torch.sin(position * div_term.unsqueeze(0))
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, scale_factor=0.5):
        super(CustomMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.depth = embed_dim // num_heads

        self.qkv_linear = nn.Linear(embed_dim, embed_dim * 3)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x.transpose(0,1)
        batch_size, seq_length, _ = x.size()

        # Linearly transform data into queries, keys, and values
        qkv = self.qkv_linear(x)
        q, k, v = torch.split(qkv, self.embed_dim, dim=2)

        # Split and transpose Q, K, V for multi-head attention
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scores = matmul_qk / (self.depth ** self.scale_factor)

        if key_padding_mask is not None:
            # Ensure the key_padding_mask is broadcastable,
            # expand it to the number of heads.
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))

        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)  # Add attn_mask to scores before softmax

        # Apply softmax to get the attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Multiply the weights by values
        weighted_average = torch.matmul(attention_weights, v)

        # Concatenate heads and apply final linear layer
        weighted_average = weighted_average.permute(0, 2, 1, 3).contiguous()
        weighted_average = weighted_average.view(batch_size, seq_length, self.embed_dim)
        output = self.out_linear(weighted_average)
        output = output.transpose(0,1)
        attn_weights_mean = torch.mean(attention_weights, dim=1)

        return output, attn_weights_mean
    
class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.cache = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, self_attn_weights = self.self_attn(tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, enc_dec_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights, enc_dec_attn_weights

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    # 【0，0，1，mask，0，0】, Attention，学每个向量上下文关系，上下文关注程度，任务作难一点，
    #  苹果 很 好吃， 苹果 手机 很好玩
    #　　　苹果 很 好吃
    # 苹果【0.5　0.1　0.4】　【0　－ｉｎｆ　－ｉｎｆ】
    #很　【0.1　0.8　0.1】　【0　0　－ｉｎｆ】
    #好吃【0.3　0.1　0.6】　【0　0　0】
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attention_weights = []
        
        for layer in self.layers:
            output, attn_weights = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attention_weights.append(attn_weights)

        return output, attention_weights

class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        self_attn_weights_list = []
        enc_dec_attn_weights_list = []

        for layer in self.layers:
            output, self_attn_weights, enc_dec_attn_weights = layer(
                output, memory, 
                tgt_mask=tgt_mask, memory_mask=memory_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
            )
            self_attn_weights_list.append(self_attn_weights)
            enc_dec_attn_weights_list.append(enc_dec_attn_weights)

        return output, self_attn_weights_list, enc_dec_attn_weights_list

class TransformerModel(nn.Module):
    def __init__(self, enc, dec, parameters, rn, new2raw_rid_dict, raw2new_rid_dict):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        d_model = parameters.hid_dim
        nhead = parameters.nhead
        nlayers = parameters.nlayers
        dropout = parameters.dropout
        ntoken = parameters.id_size
        self.pcm = parameters.pcm
        self.device = parameters.device
        self.nhead = nhead

        self.rn = rn
        self.new2raw_rid_dict = new2raw_rid_dict
        self.raw2new_rid_dict = raw2new_rid_dict


        # ntoken, d_model, nhead, nhid, nlayers, dropout=0.5
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        # self.pos_decoder = PositionalEncoding(d_model, dropout)
        # encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        encoder_layers = CustomEncoderLayer(d_model, nhead)#, d_model, dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layers, nlayers)
        # decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, d_model, dropout)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        decoder_layers = CustomDecoderLayer(d_model, nhead)#, d_model, dropout)
        self.transformer_decoder = CustomTransformerDecoder(decoder_layers, nlayers)
        spe_decoder = CustomDecoderLayer(d_model, nhead)#, d_model, dropout)
        self.spe_transformer_decoder = CustomTransformerDecoder(spe_decoder, nlayers)
        # self.encoder_id = nn.Embedding(ntoken, d_model)
        self.encoder_id = nn.Embedding(ntoken, d_model-1)
        # self.encoder_rate = nn.Linear(in_features=1, out_features=d_model//2, bias=True)
        self.encoder = enc  # Encoder
        self.d_model = d_model
        # self.decoder = nn.Linear(d_model, ntoken)
        self.decoder = dec  # DecoderMulti

        self.spe_linear = nn.Linear(d_model+1, d_model)
        spe_layers = CustomEncoderLayer(d_model, nhead)#, d_model, dropout)
        self.spe_encoder = CustomTransformerEncoder(spe_layers, nlayers)
        self.memory_attention = nn.MultiheadAttention(d_model, nhead)
        # self.spe_multi_decoder = DecoderMulti(parameters)
        # self.segment_embeddings = nn.Embedding(2, d_model)
        # nn.init.normal_(self.segment_embeddings.weight, mean=0, std=0.02)

    def generate_square_position_mask(self, sz: int, positions) -> torch.Tensor:
        """
        生成一个大小为 (sz, sz) 的遮掩张量，用于遮掩未来的位置。
        
        参数:
        - sz (int): 序列的长度。
        
        返回:
        - mask (torch.Tensor): 遮掩张量。
        """
        batch_size = positions.size(0)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        for index in range(batch_size):
            position = positions[index]
            for p in position:
                mask[index, p, [i  for i in range(p) if i not in position]] = float('-inf')
        return mask
        
    def generate_square_subsequent_mask(self, sz: int, positions) -> torch.Tensor:
        """
        生成 PCM Mask
        """
        batch_size = positions.size(0)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        for index in range(batch_size):
            mask[index, :, [p for p in positions[index] if p < sz]] = float(0.0)
        return mask
    
    def generate_square_mask(self, sz: int, positions) -> torch.Tensor:
        """
        生成一个大小为 (sz, sz) 的遮掩张量，用于遮掩未来的位置。
        
        参数:
        - sz (int): 序列的长度。
        
        返回:
        - mask (torch.Tensor): 遮掩张量。
        """
        batch_size = positions.size(0)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return mask
    
    def get_padding_mask(self, src_len):
        max_src_len = max(src_len)
        batch_size = len(src_len)
        memory_key_padding_mask  = torch.zeros(batch_size, max_src_len, dtype=torch.bool)
        for i, length in enumerate(src_len):
            if length < max_src_len:
                memory_key_padding_mask[i, length:] = True  # 屏蔽超出有效长度的位置
        return memory_key_padding_mask
        
    def forward(self, src, src_len, trg_id, trg_rate, trg_len,
                pre_grids, next_grids, constraint_mat, pro_features, 
                online_features_dict, rid_features_dict,
                teacher_forcing_ratio=0.5, random_p = 0.0):
        max_trg_len = max(trg_len)
        max_src_len = max(src_len)
        batch_size = trg_id.size(1)
        positions = src[:,:,-1].transpose(0,1)
        positions = torch.tensor([[int(i) for i in positions[index]] for index in range(batch_size)])
        src_key_padding_mask = self.get_padding_mask(src_len).to(self.device)
        src_mask = torch.full((batch_size, max_src_len, max_src_len), random_p)
        while True:
            random_mask = torch.bernoulli(src_mask).bool().to(self.device)
            if not torch.bitwise_or(random_mask, src_key_padding_mask.unsqueeze(1).expand(-1, max_src_len, -1)).all(dim=2).any():
                src_mask = random_mask
                src_mask = src_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).reshape(batch_size * self.nhead, src_mask.size(1), src_mask.size(2))
                break
        encoder_outputs = self.encoder(src)
        encoder_outputs = self.pos_embedding(encoder_outputs)
        memory, encoder_attn_weights = self.transformer_encoder(encoder_outputs, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # memory_mask = torch.full((batch_size, max_trg_len, max_src_len), random_p)
        # while True:
        #     random_mask = torch.bernoulli(memory_mask).bool().to(self.device)
        #     if not torch.bitwise_or(random_mask, src_key_padding_mask.unsqueeze(1).expand(-1, max_trg_len, -1)).all(dim=2).any():
        #         memory_mask = random_mask
        #         memory_mask = memory_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).reshape(batch_size * self.nhead, memory_mask.size(1), memory_mask.size(2))
        #         break
        PCM = self.pcm
        if PCM:
            outputs_id, outputs_rate, attention_weights = self.pcm_step(encoder_outputs, memory, src_len, src_key_padding_mask, positions, max_src_len, max_trg_len, batch_size, trg_id, trg_rate, trg_len, teacher_forcing_ratio)
            decoder_self_attn_weights, decoder_enc = attention_weights
        else:
            outputs_id, outputs_rate, attention_weights = self.normal_step(memory, src_len, positions, max_trg_len, batch_size, trg_id, trg_rate, trg_len, teacher_forcing_ratio)
        decoder_self_attn_weights, decoder_enc = attention_weights

        return outputs_id, outputs_rate, (encoder_attn_weights, decoder_self_attn_weights, decoder_enc)

    def pcm_step(self, encoder_outputs, memory, src_len, src_key_padding_mask, positions, max_src_len, max_trg_len, batch_size, trg_id, trg_rate, trg_len, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        decoder_self_attn_weights = []
        decoder_enc = []

        src_id = torch.zeros(max_src_len, batch_size, 1, dtype=int).to(self.device)
        src_rate = torch.zeros(max_src_len, batch_size, 1).to(self.device)
        for t in range(1, max_src_len):
            position_mask = self.generate_square_mask(t, positions).to(self.device)
            tgt_id = self.encoder_id(src_id[:t].squeeze(2))
            tgt = torch.cat((tgt_id, src_rate[:t]),dim=2)
            tgt = self.pos_embedding(tgt)
            output, _, _ = self.transformer_decoder(tgt, memory, tgt_mask=position_mask[:,:t,:t], tgt_key_padding_mask = src_key_padding_mask[:,:t])
            prediction_id, prediction_rate = self.decoder(output[-1:])

            top1_id = torch.argmax(prediction_id, dim=2, keepdim=True)
            new_src_id = src_id.clone()
            new_src_rate = src_rate.clone()
            new_src_id[t:t+1,:,:] = top1_id
            new_src_rate[t:t+1,:,:] = prediction_rate
            src_id = new_src_id
            src_rate = new_src_rate

        spe_id = torch.zeros(max_trg_len, batch_size, 1, dtype=int).to(self.device)
        spe_rate = torch.zeros(trg_rate.size()).to(self.device)
        spe_mask, constraint_p = self.get_spe_mask(max_trg_len, src_id, positions, src_len)
        tgt_key_padding_mask = self.get_padding_mask(trg_len).to(self.device)
        for t in range(1, max_trg_len):
            position_mask = self.generate_square_mask(t, positions).to(self.device)
            tgt_id = self.encoder_id(spe_id[:t].squeeze(2))
            tgt = torch.cat((tgt_id, spe_rate[:t]),dim=2)
            tgt = self.pos_embedding(tgt)
            output, _, _ = self.spe_transformer_decoder(tgt, memory, tgt_mask=position_mask[:,:t,:t], tgt_key_padding_mask = tgt_key_padding_mask[:,:t])
            mask_cons = self.dynamic_mask(spe_mask[t])
            prediction_id, prediction_rate = self.decoder(output[-1:], mask_cons)

            top1_id = torch.argmax(prediction_id, dim=2, keepdim=True)
            spe_mask = self.update_spe_mask(spe_mask, top1_id, t)
            new_input_id = spe_id.clone()
            new_input_rate = spe_rate.clone()
            new_input_id[t:t+1,:,:] = top1_id
            new_input_rate[t:t+1,:,:] = prediction_rate
            spe_id = new_input_id
            spe_rate = new_input_rate
        
        spe_embedding = self.encoder_id(spe_id.squeeze(2))
        spe_embedding = torch.cat((spe_embedding, spe_rate,constraint_p.to(self.device)),dim=2)
        spe_embedding = self.spe_linear(spe_embedding)
        # spe_embedding = torch.cat((spe_embedding, spe_rate),dim=2)
        # spe_embedding = spe_embedding + self.segment_embeddings(constraint_p.squeeze(2).long().to(self.device)) // no segment embedding is better
        spe_embedding = self.pos_embedding(spe_embedding)

        spe_memory, _ = self.spe_encoder(spe_embedding, mask=None, src_key_padding_mask=tgt_key_padding_mask)
        # cross_memory, _ = self.memory_attention(memory, spe_memory, spe_memory)
        # memory_combined = memory + cross_memory
        cross_memory, _ = self.memory_attention(spe_memory, memory, memory)
        memory_combined = spe_memory + cross_memory
        # memory_combined = torch.cat((memory, spe_memory), dim=0) // performance not good

        # encoder_combined = torch.cat((encoder_outputs, spe_embedding), dim=0)
        # padding_combined = torch.cat((src_key_padding_mask, tgt_key_padding_mask), dim=1)
        # memory_combined, _ = self.transformer_encoder(encoder_combined, mask=None, src_key_padding_mask=padding_combined)

        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)
        # input_id = trg_id#.clone()
        # input_rate = trg_rate#.clone()
        input_id = spe_id#.clone()
        input_rate = spe_rate#.clone()
        position_mask = self.generate_square_subsequent_mask(max_trg_len, positions).to(self.device)
        # position_mask = self.generate_square_mask(max_trg_len, positions).to(self.device)
        for t in range(1, max_trg_len):
            # position_mask = self.generate_square_mask(t, positions).to(self.device)
            # tgt_id = self.encoder_id(input_id[:t].squeeze(2))
            # tgt = torch.cat((tgt_id, input_rate[:t]),dim=2)
            tgt_id = self.encoder_id(input_id.squeeze(2))
            tgt = torch.cat((tgt_id, input_rate),dim=2)
            tgt = self.pos_embedding(tgt)
            # tgt, _, _ = self.transformer_decoder(tgt, memory_combined, tgt_mask=position_mask, tgt_key_padding_mask = tgt_key_padding_mask) // no share is better
            output, decoder_self_attn_weights_new, decoder_enc_new = self.transformer_decoder(tgt, memory_combined, tgt_mask=position_mask, tgt_key_padding_mask = tgt_key_padding_mask)
            prediction_id, prediction_rate = self.decoder(output)
            outputs_id[t] = prediction_id[t-1:t,:,:].squeeze(0)
            outputs_rate[t] = prediction_rate[t-1:t,:,:].squeeze(0)
            decoder_self_attn_weights = decoder_self_attn_weights_new
            decoder_enc = decoder_enc_new

            teacher_force = random.random() < teacher_forcing_ratio
            new_input_id = input_id.clone()
            new_input_rate = input_rate.clone()
            if not teacher_force:
                top1_id = torch.argmax(prediction_id[t-1:t,:,:], dim=2, keepdim=True)
                new_input_id[t:t+1,:,:] = top1_id
                new_input_rate[t:t+1,:,:] = prediction_rate[t-1:t,:,:]
            else:
                new_input_id[t:t+1,:,:] = trg_id[t:t+1,:,:]
                new_input_rate[t:t+1,:,:] = trg_rate[t:t+1,:,:]
            input_id = new_input_id
            input_rate = new_input_rate

        # max_trg_len, batch_size, trg_rid_size
        outputs_ids = outputs_id.permute(1, 0, 2)
        outputs_rates = outputs_rate.permute(1, 0, 2)
        for i in range(batch_size):
            outputs_ids[i][trg_len[i]:] = -100
            outputs_ids[i][trg_len[i]:, 0] = 0  # make sure argmax will return eid0
            outputs_rates[i][trg_len[i]:] = 0
        outputs_ids = outputs_ids.permute(1, 0, 2)
        outputs_rates = outputs_rates.permute(1, 0, 2)

        return outputs_ids, outputs_rates, (decoder_self_attn_weights, decoder_enc) # torch.stack(all_attention_weights, dim=0)

    def normal_step(self, memory, src_len, positions, max_trg_len, batch_size, trg_id, trg_rate, trg_len, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)
        decoder_self_attn_weights = []
        decoder_enc = []
        tgt_key_padding_mask = self.get_padding_mask(trg_len).to(self.device)
        input_id = trg_id#.clone()
        input_rate = trg_rate#.clone()
        # position constraint preidction
        for t in range(1, max_trg_len):
            position_mask = self.generate_square_mask(t, positions).to(self.device)
            tgt_id = self.encoder_id(input_id[:t].squeeze(2))
            tgt = torch.cat((tgt_id, input_rate[:t]),dim=2)
            tgt = self.pos_embedding(tgt)
            output, decoder_self_attn_weights_new, decoder_enc_new = self.transformer_decoder(tgt, memory, tgt_mask=position_mask[:,:t,:t], tgt_key_padding_mask = tgt_key_padding_mask[:,:t])
            prediction_id, prediction_rate = self.decoder(output[-1:])
            outputs_id[t] = prediction_id.squeeze(0)
            outputs_rate[t] = prediction_rate.squeeze(0)
            decoder_self_attn_weights = decoder_self_attn_weights_new
            decoder_enc = decoder_enc_new

            teacher_force = random.random() < teacher_forcing_ratio
            if not teacher_force:
                top1_id = torch.argmax(prediction_id, dim=2, keepdim=True)
                new_input_id = input_id.clone()
                new_input_rate = input_rate.clone()
                new_input_id[t:t+1,:,:] = top1_id
                new_input_rate[t:t+1,:,:] = prediction_rate
                input_id = new_input_id
                input_rate = new_input_rate

        # max_trg_len, batch_size, trg_rid_size
        outputs_ids = outputs_id.permute(1, 0, 2)
        outputs_rates = outputs_rate.permute(1, 0, 2)
        for i in range(batch_size):
            outputs_ids[i][trg_len[i]:] = -100
            outputs_ids[i][trg_len[i]:, 0] = 0  # make sure argmax will return eid0
            outputs_rates[i][trg_len[i]:] = 0
        outputs_ids = outputs_ids.permute(1, 0, 2)
        outputs_rates = outputs_rates.permute(1, 0, 2)

        return outputs_ids, outputs_rates, (decoder_self_attn_weights, decoder_enc) # torch.stack(all_attention_weights, dim=0)
    
    def get_spe_mask(self, max_trg_len, src_id, positions, src_len):
        batch_size = positions.size(0)
        # mask = torch.full((max_trg_len, batch_size, self.decoder.id_size), float('0.0')).to(self.device)
        path = [[None for i in range(batch_size)] for j in range(max_trg_len)]
        constraint_p = torch.zeros(max_trg_len, batch_size, 1)
        for i in range(batch_size):
            position = positions[i]
            ids = src_id[:,i,0]

            constraint_p[position,i,0] = 1
            for j in range(1, src_len[i]-1):
                start = position[j]
                end = position[j+1]
                start_id = ids[j].tolist()
                end_id = ids[j+1].tolist()
                path[start][i] = start_id
                path[end][i] = end_id
                if start_id !=0 and end_id != 0 and start_id != end_id:
                    try:
                        pre_u, pre_v = self.rn.edge_idx[self.new2raw_rid_dict[start_id]]
                        cur_u, cur_v = self.rn.edge_idx[self.new2raw_rid_dict[end_id]]
                        route = nx.astar_path(self.rn, pre_v, cur_u)
                        if route is not None and len(route) > 1:
                            route_s = self.p2s(route)
                        else:
                            route_s = []

                        if len(route_s) == 0 or start_id != route_s[0]:
                            route_s = [start_id] + route_s
                        if len(route_s) == 0 or end_id != route_s[-1]:
                            route_s = route_s + [end_id]
                        for r_i in range(start+1,end):
                            path[r_i][i] = route_s
                    except:
                        pass
                else:
                    pass
        return path,constraint_p
    
    def update_spe_mask(self, mask, pred, t):
        if t >= len(mask) -1:
            return mask
        mask_pre = mask[t]
        mask_next = mask[t+1]
        for i in range(len(mask_pre)):
            p = pred[0][i][0].tolist()
            if type(mask_pre[i]) == list and type(mask_next[i]) == list:
                mask[t+1][i] = mask[t+1][i][mask_next[i].index(p):]       
        return mask

    def p2s(self, p):
        s = []
        for i in range(len(p)-1):
            eid_raw = self.rn[p[i]][p[i+1]]['eid']
            eid_new = self.raw2new_rid_dict[eid_raw]
            if len(s) == 0 or eid_new != s[-1]:
                s.append(eid_new)
        return s

    def dynamic_mask(self, ids):
        batch_size = len(ids)
        mask = torch.full((1, batch_size, self.decoder.id_size), float('0.0')).to(self.device)
        for i in range(batch_size):
            m = ids[i]
            if m is None:
                continue
            if type(m) == int:
                mask[0, i, :m] = float('-inf')
                mask[0, i, m+1:] = float('-inf')
            elif type(m) == list:
                try:
                    mask[0, i, :] = float('-inf')
                    mask[0, i, m] = float('0.0')
                except:
                    pdb.set_trace()
        return mask