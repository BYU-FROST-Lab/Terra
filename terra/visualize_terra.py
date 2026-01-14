from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl
import numpy as np

import networkx as nx
import open3d as o3d

from utils import numeric_key

class Terra_Visualizer():
    def __init__(self, level_offset, terrain_colors=None):
        self.level_offset = level_offset
        if terrain_colors:
            self.num_terrains = len(terrain_colors)
            self.terrain_colors = terrain_colors
        else:
            self.num_terrains = 3
            self.terrain_colors = [[1,0,0],[0,1,0],[0,0,1]]
    
    def display_3dsg(self, G, node_colors=None, pc=None):
        geometries = []
        nodes = []
        points = []
        lines = []
        colors = []
        node_idx_map = {}
        for i, n_id in enumerate(list(G.nodes)):
            level_num = G.nodes[n_id]["level"] + 1
            z = level_num * self.level_offset
            xy = G.nodes[n_id]["pos"]
            
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5).translate([xy[0],xy[1],z])
            if node_colors is not None:
                if n_id in node_colors:
                    sphere.paint_uniform_color(node_colors[n_id])
                else:
                    sphere.paint_uniform_color([0,0,0])
            elif G.nodes[n_id]["terrain_id"] > -1:
                sphere.paint_uniform_color(self.terrain_colors[G.nodes[n_id]["terrain_id"]])
            else:
                sphere.paint_uniform_color([0,0,0])
            nodes.append(sphere)
            points.append([xy[0], xy[1], z])
            
            node_idx_map[n_id] = i
        geometries.extend(nodes)
        
        for (u,v) in list(G.edges()):
            if u in node_idx_map and v in node_idx_map:
                lines.append([node_idx_map[u], node_idx_map[v]])
                colors.append([0.5,0.5,0.5])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
        geometries.extend([line_set])
        
        if pc is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd.paint_uniform_color([0.5,0.5,0.5])
            geometries.extend([pcd])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geo in geometries:
            vis.add_geometry(geo)
        render_opt = vis.get_render_option()
        render_opt.point_size = 3.0  # smaller points
        vis.run()
        vis.destroy_window()
        
        # o3d.visualization.draw_geometries(geometries)
        # return geometries


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--terra_3dsg',
                        type=str,
                        help='Filepath for terra_3dsg saved from build_terra.py')
    parser.add_argument('--global_pc',
                        type=str,
                        help='Directory where global point clouds are saved')
    args = parser.parse_args()
    
    global_pc_files = sorted(Path(args.global_pc).glob("*.npy"),key=numeric_key)        
    latest_global_pc_file = global_pc_files[-1] # global_map_idx 33 for sunny midday data
    global_pc = np.load(latest_global_pc_file) # (num_pts,4)
    
    with open(args.terra_3dsg, "rb") as f:
        terra_3dsg = pkl.load(f)
    
    tv = Terra_Visualizer(level_offset=25)
    
    # Display Regions 
    region_subgraph = terra_3dsg.subgraph(
        [n_id for n_id in list(terra_3dsg.nodes) if terra_3dsg.nodes[n_id]["level"] > 1]
    )
    tv.display_3dsg(region_subgraph)
    
    # Display Places
    places_subgraph = terra_3dsg.subgraph(
        [n_id for n_id in list(terra_3dsg.nodes) if terra_3dsg.nodes[n_id]["level"] == 1]
    )
    tv.display_3dsg(places_subgraph)
    
    # Display full 3DSG with point cloud
    tv.display_3dsg(terra_3dsg)
    tv.display_3dsg(terra_3dsg,pc=global_pc[:,:3])