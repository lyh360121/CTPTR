from common.road_network import load_rn_shp
import json

import pdb


if __name__=='__main__':
    rn_dir = "./data/map/road_network/"

    rn = load_rn_shp(rn_dir, is_directed=True)

    # store extra dict
    raw_rn_dict = {}
    rn_dict = {}
    new2raw = {}
    raw2new = {}
    new_id = 1
    for edge in rn.edges:
        edge_info = rn.edges[edge]
        edge_info = {key: value for key, value in edge_info.items()}
        edge_info['coords'] = [[point.lat, point.lng] for point in edge_info['coords']]
        raw_rn_dict[str(edge_info['eid'])] = edge_info
        rn_dict[str(new_id)] = edge_info
        new2raw[str(new_id)] = str(edge_info['eid'])
        raw2new[str(edge_info['eid'])] = str(new_id)
        new_id +=1
    
    with open('./data/map/extra_info/raw_rn_dict.json', 'w') as f:
        json.dump(raw_rn_dict, f)

    with open('./data/map/extra_info/rn_dict.json', 'w') as f:
        json.dump(rn_dict, f)   
    
    with open('./data/map/extra_info/new2raw_rid.json', 'w') as f:
        json.dump(new2raw, f)

    with open('./data/map/extra_info/raw2new_rid.json', 'w') as f:
        json.dump(raw2new, f) 