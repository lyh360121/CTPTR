from utils.utils import create_dir
from utils.save_traj import SaveTraj2MM
from common.road_network import load_rn_shp
from tqdm import tqdm
import os
import numpy as np
import random
import pdb

from datetime import datetime
from common.trajectory import Trajectory, STPoint
from common.spatial_func import project_pt_to_segment
from map_matching.candidate_point import CandidatePoint

class ParseTraj:
    """
    ParseTraj is an abstract class for parsing trajectory.
    It defines parse() function for parsing trajectory.
    """
    def __init__(self):
        pass

    def parse(self, input_path):
        """
        The parse() function is to load data to a list of Trajectory()
        """
        pass


class ParseMMTraj(ParseTraj):
    """
    Parse map matched GPS points to trajectories list. No extra data preprocessing
    """
    def __init__(self, rn):
        super().__init__()
        self.rn = rn

    def parse(self, input_path):
        """
        Args:
        -----
        input_path:
            str. input directory with file name
        Returns:
        --------
        trajs:
            list. list of trajectories. trajs contain input_path file's all gps points
        """
        time_format = '%Y/%m/%d %H:%M:%S'
        tid_to_remove = '[:/ ]'
        with open(input_path, 'r') as f:
            trajs = []
            pt_list = []
            for line in f.readlines():
                attrs = line.rstrip().split(',')
                if attrs[0] == '#':
                    if len(pt_list) > 1:
                        traj = Trajectory(oid, tid, pt_list)
                        trajs.append(traj)
                    oid = attrs[2]
                    tid = attrs[1]
                    pt_list = []
                else:
                    lat = float(attrs[1])
                    lng = float(attrs[2])
                    if attrs[3] == 'None':
                        candi_pt = None
                    else:
                        eid = int(attrs[3])
                        proj_lat = float(attrs[4])
                        proj_lng = float(attrs[5])
                        error = float(attrs[6])
                        offset = float(attrs[7])
                        # rate = float(attrs[8])
                        edge = rn.edge_idx[eid]
                        u, v = edge
                        coords = rn[u][v]['coords']
                        raw_pt = STPoint(lat, lng, datetime.strptime(attrs[0], time_format))
                        candidates = [project_pt_to_segment(coords[i], coords[i + 1], raw_pt) for i in range(len(coords) - 1)]
                        idx, (projection, rate, dist) = min(enumerate(candidates), key=lambda v: v[1][2])
                        print(rate, '=============')
                        candi_pt = CandidatePoint(proj_lat, proj_lng, eid, error, offset, rate)
                    pt = STPoint(lat, lng, datetime.strptime(attrs[0], time_format), {'candi_pt': candi_pt})
                    # pt contains all the attributes of class STPoint
                    pt_list.append(pt)
            if len(pt_list) > 1:
                traj = Trajectory(oid, tid, pt_list)
                trajs.append(traj)
        return trajs


if __name__ == '__main__':
    """
    split original data to train, valid and test datasets
    """
    traj_input_dir = "./data/raw_trajectory/"
    output_dir = "./data/model_data/"
    rn_dir = "./data/map/road_network/"

    create_dir(output_dir)
    train_data_dir = output_dir + 'train_data/'
    create_dir(train_data_dir)
    val_data_dir = output_dir + 'valid_data/'
    create_dir(val_data_dir)
    test_data_dir = output_dir + 'test_data/'
    create_dir(test_data_dir)

    rn = load_rn_shp(rn_dir, is_directed=True)
    trg_parser = ParseMMTraj(rn)
    trg_saver = SaveTraj2MM()

    for file_name in tqdm(os.listdir(traj_input_dir)):
        traj_input_path = os.path.join(traj_input_dir, file_name)
        trg_trajs = np.array(trg_parser.parse(traj_input_path))
        ttl_lens = len(trg_trajs)
        test_inds = random.sample(range(ttl_lens), int(ttl_lens * 0.1))  # 10% as test data
        tmp_inds = [ind for ind in range(ttl_lens) if ind not in test_inds]
        val_inds = random.sample(tmp_inds, int(ttl_lens * 0.2))  # 20% as validation data
        train_inds = [ind for ind in tmp_inds if ind not in val_inds]  # 70% as training data

        trg_saver.store(trg_trajs[train_inds], os.path.join(train_data_dir, 'train_' + file_name))
        # print("target traj train len: ", len(trg_trajs[train_inds]))
        trg_saver.store(trg_trajs[val_inds], os.path.join(val_data_dir, 'val_' + file_name))
        # print("target traj val len: ", len(trg_trajs[val_inds]))
        trg_saver.store(trg_trajs[test_inds], os.path.join(test_data_dir, 'test_' + file_name))
        # print("target traj test len: ", len(trg_trajs[test_inds]))