from argparse import ArgumentParser
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import open3d as o3d

from terra.utils import numeric_key, random_color, find_latest_itr, find_latest_file

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

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--data_folder', 
                        type=str, 
                        default="/docker_ros2_ws/src/oasis2/data/south_campus_4_21_2025", 
                        help='Directory to where folders of lidar and global scans were saved')
    parser.add_argument('--output_folder', 
                        type=str, 
                        default="/docker_ros2_ws/src/oasis2/data/south_campus_4_21_2025/output", 
                        help='Directory to where MSMap data was saved')
    parser.add_argument('--num_terrain',
                        type=int,
                        default=3,
                        help="Number of terrain classes used by YOLO model")
    parser.add_argument('--pt_size',
                        type=float,
                        default=2.0,
                        help="Size of points to display")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    
    global_pc_folder = os.path.join(args.data_folder, "global_pc")
    global_pc_files = sorted(Path(global_pc_folder).glob("*.npy"),key=numeric_key)        
    latest_global_pc_file = global_pc_files[-1] # global_map_idx 33 for sunny midday data
    global_pc = np.load(latest_global_pc_file) # (num_pts,4)
    
    latest_file = find_latest_file(args.output_folder)
    last_itr = find_latest_itr(args.output_folder)
    pc_clip_dict_path = os.path.join(args.output_folder, latest_file)
    print(pc_clip_dict_path)
    with open(pc_clip_dict_path, "rb") as f: pc_clip_dict = pkl.load(f)
    
    clipid_2_globalpts = map_clipid_to_globalpts(global_pc, pc_clip_dict)
    
    # Display Global Point Cloud colored by CLIP IDs
    cmap = plt.get_cmap("tab10")  # 10 distinct colors
    chosen_colors = [cmap(i % 10)[:3] for i in range(args.num_terrain)]
    pcds = []
    terrain_pcds = []
    for clip_id in clipid_2_globalpts.keys():
        pcd = o3d.geometry.PointCloud()
        if clip_id == -1:
            pcd.points = o3d.utility.Vector3dVector(global_pc[clipid_2_globalpts[clip_id],:3])
            pcd.paint_uniform_color([0.5,0.5,0.5])
        elif clip_id < args.num_terrain:
            pcd.points = o3d.utility.Vector3dVector(global_pc[clipid_2_globalpts[clip_id], :3])
            pcd.paint_uniform_color(chosen_colors[clip_id])
            terrain_pcds.append(pcd)
        else:
            random_col = random_color()
            pcd.points = o3d.utility.Vector3dVector(global_pc[clipid_2_globalpts[clip_id], :3])
            pcd.paint_uniform_color(random_col)
        pcds.append(pcd)
        
    # o3d.visualization.draw_geometries(pcds)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pcd in terrain_pcds:
        vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = args.pt_size  # smaller points
    vis.run()
    vis.destroy_window()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pcd in pcds:
        vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = args.pt_size  # smaller points
    vis.run()
    vis.destroy_window()