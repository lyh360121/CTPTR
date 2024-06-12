 #!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/10/29 17:37

from models.model_utils import toseq, get_constraint_mask, rate2gps
from models.loss_fn import cal_id_acc, check_rn_dis_loss

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pdb


def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)


def plot_attention(attention, input_seq, output_seq):
    """
    Plots a heatmap of attention weights.
    
    :param attention: A 2D numpy array of attention weights.
    :param input_seq: List of input tokens.
    :param output_seq: List of output tokens.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_seq, rotation=90)
    ax.set_yticklabels([''] + output_seq)

    # Show label at every tick
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.show()


def train(model, iterator, optimizer, log_vars, rn_dict, grid_rn_dict, rn,
          raw2new_rid_dict, online_features_dict, rid_features_dict, parameters):
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()

    epoch_ttl_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    device = parameters.device

    # num = len(iterator) - 1
    # lr = optimizer.param_groups[0]['lr']
    # mult = (10.0 / 1e-8) ** (1/num)
    # best_loss = 0.
    # losses = []
    # log_lrs = []

    for i, batch in enumerate(iterator):
        src_grid_seqs, src_gps_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths = batch
        if parameters.dis_prob_mask_flag:
            constraint_mat, pre_grids, next_grids = get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths,
                                                                        trg_lengths, grid_rn_dict, rn, raw2new_rid_dict,
                                                                        parameters)
            constraint_mat = constraint_mat.permute(1, 0, 2).to(device)
            pre_grids = pre_grids.permute(1, 0, 2).to(device)
            next_grids = next_grids.permute(1, 0, 2).to(device)
        else:
            max_trg_len = max(trg_lengths)
            batch_size = src_grid_seqs.size(0)
            constraint_mat = torch.zeros(max_trg_len, batch_size, parameters.id_size, device=device)
            pre_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)
            next_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)

        src_pro_feas = src_pro_feas.float().to(device)

        src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
        trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
        trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
        trg_rates = trg_rates.permute(1, 0, 2).to(device)

        # constraint_mat = [trg len, batch size, id size]
        # src_grid_seqs = [src len, batch size, 2] [6,128,3]
        # src_lengths = [batch size] [128]
        # trg_gps_seqs = [trg len, batch size, 2]
        # trg_rids = [trg len, batch size, 1] [33, 128,1]
        # trg_rates = [trg len, batch size, 1]
        # trg_lengths = [batch size]

        optimizer.zero_grad()
        output_ids, output_rates, attention_weights = model(src_grid_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                         pre_grids, next_grids, constraint_mat, src_pro_feas,
                                         online_features_dict, rid_features_dict, parameters.tf_ratio)
        output_rates = output_rates.squeeze(2)
        trg_rids = trg_rids.squeeze(2)
        trg_rates = trg_rates.squeeze(2)
        # attention_weights_array = attention_weights.squeeze(1).permute(1, 0, 2).cpu().detach().numpy()
        # pdb.set_trace()
        # plot_attention(attention_weights_array[0],  list(range(len(src_grid_seqs.permute(1,0,2)[0]))), list(range(len(trg_rids.permute(1,0)[0]))))

        # output_ids = [trg len, batch size, id one hot output dim]
        # output_rates = [trg len, batch size]
        # trg_rids = [trg len, batch size]
        # trg_rates = [trg len, batch size]

        # rid loss, only show and not bbp
        loss_ids1, recall, precision = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths)

        # for bbp
        rate_mask = (torch.argmax(output_ids, dim=2) == trg_rids).float()
        output_ids_dim = output_ids.shape[-1]
        output_ids = output_ids[1:].reshape(-1, output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
        trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
        # view size is not compatible with input tensor's size and stride ==> use reshape() instead

        loss_train_ids = criterion_ce(output_ids, trg_rids)
        # points_pred = []
        # points_trg = []
        # p_a = output_ids.argmax(1)
        # p_b = output_rates[1:].reshape(-1)
        # p_c = trg_rids
        # p_d = trg_rates[1:].reshape(-1)
        # for pi in range(trg_rids.size(0)):
        #     point_pred = rate2gps(rn_dict, p_a[pi], p_b[pi], parameters)
        #     point_trg = rate2gps(rn_dict, p_c[pi], p_d[pi], parameters)
        #     points_pred.append([point_pred.lat, point_pred.lng])
        #     points_trg.append([point_trg.lat, point_trg.lng])
        # loss_rates = criterion_reg(torch.tensor(points_pred), torch.tensor(points_trg))
        # loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
        loss_rates_1 = criterion_reg(output_rates[1:]*rate_mask[1:], trg_rates[1:]*rate_mask[1:])
        loss_rates_2 = criterion_reg(torch.zeros(rate_mask[1:].shape).to(device), 1-rate_mask[1:])
        loss_rates =(loss_rates_1 + loss_rates_2) * parameters.lambda1
        ttl_loss = loss_train_ids + loss_rates
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()

        epoch_ttl_loss += ttl_loss.item()
        epoch_id1_loss += loss_ids1
        epoch_recall_loss += recall
        epoch_precision_loss += precision
        # epoch_train_id_loss += loss_train_ids.item()
        epoch_rate_loss += loss_rates.item()

    #     # 更新学习率
    #     lr *= mult
    #     optimizer.param_groups[0]['lr'] *= lr
    #     smoothed_loss = ttl_loss / (1 - 0.98**(i+1))
    #     losses.append(smoothed_loss)
    #     log_lrs.append(lr)
    #     if i > 0 and smoothed_loss > 4 * best_loss:
    #         break
    #     if smoothed_loss < best_loss or i == 0:
    #         best_loss = smoothed_loss
    # pdb.set_trace()
    return log_vars, epoch_ttl_loss / len(iterator), epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
           epoch_precision_loss / len(iterator), epoch_rate_loss / len(iterator), epoch_train_id_loss / len(iterator)


def evaluate(model, iterator, rn_dict, grid_rn_dict, rn, raw2new_rid_dict,
             online_features_dict, rid_features_dict, raw_rn_dict, new2raw_rid_dict, parameters):
    model.eval()  # must have this line since it will affect dropout and batch normalization

    epoch_dis_mae_loss = 0
    epoch_dis_rmse_loss = 0
    epoch_dis_rn_mae_loss = 0
    epoch_dis_rn_rmse_loss = 0
    epoch_id1_loss = 0
    epoch_recall_loss = 0
    epoch_precision_loss = 0
    epoch_rate_loss = 0
    epoch_id_loss = 0 # loss from dl model
    device = parameters.device
    criterion_ce = nn.NLLLoss()
    criterion_reg = nn.MSELoss()

    with torch.no_grad():  # this line can help speed up evaluation
        for i, batch in enumerate(iterator):
            src_grid_seqs, src_gps_seqs, src_pro_feas, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths = batch

            if parameters.dis_prob_mask_flag:
                constraint_mat, pre_grids, next_grids = get_constraint_mask(src_grid_seqs, src_gps_seqs, src_lengths,
                                                                            trg_lengths, grid_rn_dict, rn,
                                                                            raw2new_rid_dict, parameters)
                constraint_mat = constraint_mat.permute(1, 0, 2).to(device)
                pre_grids = pre_grids.permute(1, 0, 2).to(device)
                next_grids = next_grids.permute(1, 0, 2).to(device)
            else:
                max_trg_len = max(trg_lengths)
                batch_size = src_grid_seqs.size(0)
                constraint_mat = torch.zeros(max_trg_len, batch_size, parameters.id_size).to(device)
                pre_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)
                next_grids = torch.zeros(max_trg_len, batch_size, 3).to(device)

            src_pro_feas = src_pro_feas.float().to(device)

            src_grid_seqs = src_grid_seqs.permute(1, 0, 2).to(device)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2).to(device)
            trg_rids = trg_rids.permute(1, 0, 2).long().to(device)
            trg_rates = trg_rates.permute(1, 0, 2).to(device)

            # constraint_mat = [trg len, batch size, id size]
            # src_grid_seqs = [src len, batch size, 2]
            # src_pro_feas = [batch size, feature dim]
            # src_lengths = [batch size]
            # trg_gps_seqs = [trg len, batch size, 2]
            # trg_rids = [trg len, batch size, 1]
            # trg_rates = [trg len, batch size, 1]
            # trg_lengths = [batch size]

            output_ids, output_rates, attention_weights = model(src_grid_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                             pre_grids, next_grids, constraint_mat,
                                             src_pro_feas, online_features_dict, rid_features_dict,
                                             teacher_forcing_ratio=0)

            # plot_attention(attention_weights[0][0][0].cpu(), list(range(max(src_lengths))), list(range(max(trg_lengths))))
            output_rates = output_rates.squeeze(2)
            output_seqs = toseq(rn_dict, output_ids, output_rates, parameters)
            trg_rids = trg_rids.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            # output_ids = [trg len, batch size, id one hot output dim]
            # output_rates = [trg len, batch size]
            # trg_rids = [trg len, batch size]
            # trg_rates = [trg len, batch size]

            # rid loss, only show and not bbp
            loss_ids1, recall, precision = cal_id_acc(output_ids[1:], trg_rids[1:], trg_lengths, debug=True)
            # distance loss
            dis_mae_loss, dis_rmse_loss, dis_rn_mae_loss, dis_rn_rmse_loss = check_rn_dis_loss(output_seqs[1:],
                                                                                               output_ids[1:],
                                                                                               output_rates[1:],
                                                                                               trg_gps_seqs[1:],
                                                                                               trg_rids[1:],
                                                                                               trg_rates[1:],
                                                                                               trg_lengths,
                                                                                               rn, raw_rn_dict,
                                                                                               new2raw_rid_dict)

            # for bbp
            rate_mask = (torch.argmax(output_ids, dim=2) == trg_rids).float()
            output_ids_dim = output_ids.shape[-1]
            output_ids = output_ids[1:].reshape(-1,
                                                output_ids_dim)  # [(trg len - 1)* batch size, output id one hot dim]
            trg_rids = trg_rids[1:].reshape(-1)  # [(trg len - 1) * batch size],
            loss_ids = criterion_ce(output_ids, trg_rids)
            # rate loss
            loss_rates_1 = criterion_reg(output_rates[1:]*rate_mask[1:], trg_rates[1:]*rate_mask[1:])
            loss_rates_2 = criterion_reg(torch.zeros(rate_mask[1:].shape).to(device), 1-rate_mask[1:])
            loss_rates =(loss_rates_1 + loss_rates_2) * parameters.lambda1
            # loss_rates = criterion_reg(output_rates[1:], trg_rates[1:]) * parameters.lambda1
            # loss_rates.size = [(trg len - 1), batch size], --> [(trg len - 1)* batch size,1]

            epoch_dis_mae_loss += dis_mae_loss
            epoch_dis_rmse_loss += dis_rmse_loss
            epoch_dis_rn_mae_loss += dis_rn_mae_loss
            epoch_dis_rn_rmse_loss += dis_rn_rmse_loss
            epoch_id1_loss += loss_ids1
            epoch_recall_loss += recall
            epoch_precision_loss += precision
            epoch_rate_loss += loss_rates.item()
            epoch_id_loss += loss_ids.item()

        return epoch_id1_loss / len(iterator), epoch_recall_loss / len(iterator), \
               epoch_precision_loss / len(iterator), \
               epoch_dis_mae_loss / len(iterator), epoch_dis_rmse_loss / len(iterator), \
               epoch_dis_rn_mae_loss / len(iterator), epoch_dis_rn_rmse_loss / len(iterator), \
               epoch_rate_loss / len(iterator), epoch_id_loss / len(iterator)
