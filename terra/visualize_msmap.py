from argparse import ArgumentParser
import os
from tqdm import tqdm
import time
from pathlib import Path
import numpy as np
import pickle as pkl
from scipy.spatial import KDTree
from sklearn.cluster import SpectralClustering
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion

import networkx as nx
import clip
import open3d as o3d

def map_clipid_to_globalpts(global_pc, pc_clip_dict):
    count_threshold = 2
    clipid_2_globalpts = {}
    for global_idx in range(global_pc.shape[0]):
        if global_idx in pc_clip_dict.keys():
            max_id, max_count = max(pc_clip_dict[global_idx].items(), key=lambda x: x[1])
            # Make sure max_count is more than some threshold
            if max_count < count_threshold:
                if -1 in clipid_2_globalpts.keys():
                    clipid_2_globalpts[-1].append(global_idx)
                else:
                    clipid_2_globalpts[-1] = [global_idx]    
            elif max_id in clipid_2_globalpts.keys():
                clipid_2_globalpts[max_id].append(global_idx)
            else:
                clipid_2_globalpts[max_id] = [global_idx]
        else:
            if -1 in clipid_2_globalpts.keys():
                clipid_2_globalpts[-1].append(global_idx)
            else:
                clipid_2_globalpts[-1] = [global_idx]
    return clipid_2_globalpts

def random_color():
    return np.random.rand(3).tolist()  # Generates a random RGB color

import re
def numeric_key(path):
    match = re.search(r"(\d+\.\d+)", path.stem)  # grabs the float in the name
    return float(match.group()) if match else float('inf')

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--data_folder', 
                        type=str, 
                        default="/docker_ros2_ws/src/oasis2/data/south_campus_4_21_2025", 
                        help='Directory to where folders of local and global scans were saved')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    data_folder = args.data_folder
    
    global_pc_folder = os.path.join(data_folder, "global_pc")
    global_pc_files = sorted(Path(global_pc_folder).glob("*.npy"),key=numeric_key)        
    latest_global_pc_file = global_pc_files[-1] # global_map_idx 33 for sunny midday data
    global_pc = np.load(latest_global_pc_file) # (num_pts,4)
    
    pc_clip_dict_path = os.path.join(data_folder, "ptxpt_pc_dict_itr2090.pkl")
    with open(pc_clip_dict_path, "rb") as f: pc_clip_dict = pkl.load(f)
    
    clipid_2_globalpts = map_clipid_to_globalpts(global_pc, pc_clip_dict)
    
    # Display Global Point Cloud colored by CLIP IDs
    pcds = []
    for clip_id in clipid_2_globalpts.keys():
        pcd = o3d.geometry.PointCloud()
        if clip_id == -1:
            pcd.points = o3d.utility.Vector3dVector(global_pc[clipid_2_globalpts[clip_id],:3])
            pcd.paint_uniform_color([0.5,0.5,0.5])
        else:
            random_col = random_color()
            pcd.points = o3d.utility.Vector3dVector(global_pc[clipid_2_globalpts[clip_id], :3])
            pcd.paint_uniform_color(random_col)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)