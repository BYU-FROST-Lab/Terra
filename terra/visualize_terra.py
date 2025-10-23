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

def display_3dsg_with_o3d(scenegraph, level_offset=10, node_colors=None):
    geometries = []
    terrain_colors = [[1,0,0],[0,1,0],[0,0,1],[1,0,1]]
    nodes = []
    points = []
    lines = []
    colors = []
    node_idx_map = {}
    for i, n_id in enumerate(list(scenegraph.nodes)):
        level_num = scenegraph.nodes[n_id]["level"] + 1
        z = level_num * level_offset
        xy = scenegraph.nodes[n_id]["pos"]
        text_label = o3d.t.geometry.TriangleMesh.create_text(str(n_id), depth=0.8).to_legacy()
        text_label.paint_uniform_color([0, 0, 0])  # or any color
        text_label.transform([[0.1, 0, 0, xy[0]], [0, 0.1, 0, xy[1]], [0, 0, 0.1, z+2],
                                    [0, 0, 0, 1]])  
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5).translate([xy[0],xy[1],z])
        if node_colors:
            if n_id in node_colors:
                sphere.paint_uniform_color(node_colors[n_id])
            else:
                sphere.paint_uniform_color([0,0,0])
        elif scenegraph.nodes[n_id]["terrain_id"] > -1:
            sphere.paint_uniform_color(terrain_colors[scenegraph.nodes[n_id]["terrain_id"]])
        else:
            sphere.paint_uniform_color([0,0,0])
        nodes.append(text_label)
        nodes.append(sphere)
        points.append([xy[0], xy[1], z])
        
        node_idx_map[n_id] = i
    geometries.extend(nodes)
    
    for (u,v) in list(scenegraph.edges()):
        if u in node_idx_map and v in node_idx_map:
            lines.append([node_idx_map[u], node_idx_map[v]])
            colors.append([0.5,0.5,0.5])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    geometries.extend([line_set])
        
    return geometries

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
    parser.add_argument('--terra_filename', 
                        type=str, 
                        default="terra_3dsg.pkl", 
                        help='Filename of 3DSG saved from build_terrain_hierregion_3dsg.py')
    parser.add_argument('--level_offset',type=float,default=10,help="How far apart to space 3DSG layers (in m)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    ###################################
    ## Load Data and Semantic Models ##
    ###################################
    args = arg_parser()
    data_folder = args.data_folder
    terra_3dsg = args.terra_3dsg
    
    global_pc_folder = os.path.join(data_folder, "global_pc")
    global_pc_files = sorted(Path(global_pc_folder).glob("*.npy"),key=numeric_key)        
    latest_global_pc_file = global_pc_files[-1] # global_map_idx 33 for sunny midday data
    global_pc = np.load(latest_global_pc_file) # (num_pts,4)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()
    
    ## Load 3DSG Terrain and Region-Based
    with open(data_folder+"/"+terra_3dsg, "rb") as f:
        TERRA_3DSG = pkl.load(f)
    
    print("Loaded 3DSG Terrain+Region Layers:",TERRA_3DSG)
    
    geo = display_3dsg_with_o3d(TERRA_3DSG, args.level_offset)
    o3d.visualization.draw_geometries(geo)
    
    # # # Get region and place node subgraphs
    # region_subgraph = TERRA_3DSG.subgraph(
    #     [n_id for n_id in list(TERRA_3DSG.nodes) if TERRA_3DSG.nodes[n_id]["level"] > 1]
    # )
    # # reg_geo = display_3dsg_with_o3d(region_subgraph, level_offset)
    # # o3d.visualization.draw_geometries(reg_geo)
    
    # places_subgraph = TERRA_3DSG.subgraph(
    #     [n_id for n_id in list(TERRA_3DSG.nodes) if TERRA_3DSG.nodes[n_id]["level"] == 1]
    # )
    # # places_geo = display_3dsg_with_o3d(places_subgraph, level_offset)
    # # o3d.visualization.draw_geometries(places_geo)
    
    
    # print("Loading saved CLIP semantic results...")
    # global_clip_path = os.path.join(data_folder, "global_clip_filt.pt")
    semantic_gidxs_path = os.path.join(data_folder, "semantic_gidxs.pkl")
    terrain_gidx_path = os.path.join(data_folder, "terrain_gidx.pkl")
    nonterrain_gidx_path = os.path.join(data_folder, "nonterrain_gidx.pkl")
    nonsemantic_gidx_path = os.path.join(data_folder, "nonsemantic_gidx.pkl")
    terrain_ids_path = os.path.join(data_folder, "terrain_ids.pkl")
    # global_clip_filt = torch.load(global_clip_path)
    with open(semantic_gidxs_path, "rb") as f: semantic_gidxs = pkl.load(f)
    with open(terrain_gidx_path, "rb") as f: terrain_gidxs = pkl.load(f)
    with open(nonterrain_gidx_path, "rb") as f: nonterrain_gidxs = pkl.load(f)
    with open(nonsemantic_gidx_path, "rb") as f: nonsemantic_gidxs = pkl.load(f)
    with open(terrain_ids_path, "rb") as f: terrain_ids = pkl.load(f)
    # # print(f"{len(terrain_gidx)} terrain points out of {global_pc.shape[0]}")
    # # print(f"{len(nonterrain_gidx)} nonterrain semantic points out of {global_pc.shape[0]}")
    # # print(f"{len(nonsemantic_gidx)} non-semantic points out of {global_pc.shape[0]}")
    
    pcd_sidewalk = o3d.geometry.PointCloud()
    pcd_sidewalk.points = o3d.utility.Vector3dVector(global_pc[terrain_ids[0],:3])#[terrain_gidx,:3])
    pcd_sidewalk.paint_uniform_color([0.55,0.55,0.55])
    
    pcd_grass = o3d.geometry.PointCloud()
    pcd_grass.points = o3d.utility.Vector3dVector(global_pc[terrain_ids[1],:3])#[terrain_gidx,:3])
    pcd_grass.paint_uniform_color([0.4,0.4,0.4])
    
    pcd_asphalt = o3d.geometry.PointCloud()
    pcd_asphalt.points = o3d.utility.Vector3dVector(global_pc[terrain_ids[2],:3])#[terrain_gidx,:3])
    pcd_asphalt.paint_uniform_color([0.7,0.7,0.7])
    
    pcd_nonterrain = o3d.geometry.PointCloud()
    pcd_nonterrain.points = o3d.utility.Vector3dVector(global_pc[nonterrain_gidxs,:3])#[terrain_gidx,:3])
    pcd_nonterrain.paint_uniform_color([0.25,0.25,0.25])
    
    o3d.visualization.draw_geometries(geo+[pcd_sidewalk, pcd_grass, pcd_asphalt])
    
    o3d.visualization.draw_geometries(geo+[pcd_sidewalk, pcd_grass, pcd_asphalt, pcd_nonterrain])
    