from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import open3d as o3d

from utils import numeric_key
from terra_utils import copy_obb

class TerraVisualizer():
    def __init__(self, level_offset, terrain_colors=None):
        self.level_offset = level_offset
        if terrain_colors:
            self.num_terrains = len(terrain_colors)
            self.terrain_colors = terrain_colors
        else:
            self.num_terrains = 3
            self.terrain_colors = [[1,0,0],[0,1,0],[0,0,1]]
        self.grays = [[0.3,0.3,0.3],[0.5,0.5,0.5],[0.75,0.75,0.75],[0.1,0.1,0.1]]
           
    def display_places(self, G):
        places_subgraph = G.subgraph(
            [n_id for n_id in list(G.nodes) if G.nodes[n_id]["level"] == 1]
        )
        self.display_3dsg(places_subgraph)
    
    def display_regions(self, G):
        regions_subgraph = G.subgraph(
            [n_id for n_id in list(G.nodes) if G.nodes[n_id]["level"] > 1]
        )
        self.display_3dsg(regions_subgraph)
    
    def display_3dsg(self, G, node_colors=None, pc=None, return_geo=False):
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
                colors.append([0.75,0.75,0.75])
        
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
        
        if return_geo:
            return geometries
        else:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for geo in geometries:
                vis.add_geometry(geo)
            render_opt = vis.get_render_option()
            render_opt.point_size = 2.0
            vis.run()
            vis.destroy_window()

    def display_terra(self, terra, display_pc=False):
        geo_3dsg = self.display_3dsg(terra.terra_3dsg, return_geo=True)
        
        # add in bounding boxes for nodes
        num_tasks = len(set(t.get_task_idx() for t in terra.objects))
        if num_tasks <= 10:
            cmap = plt.get_cmap("tab10")
        else:
            cmap = plt.get_cmap("tab20")
        task_colors = {task_idx: cmap(i % 20)[:3] for i, task_idx in enumerate(range(num_tasks))}
        for tobj in terra.objects:
            bbox = copy_obb(tobj.get_bbox())
            z_init = bbox.center[2]
            bbox.translate([0, 0, self.level_offset-z_init], relative=True)
            task_idx = tobj.get_task_idx()
            color = task_colors[task_idx]
            bbox.color = (0,0,0)#color  # OrientedBoundingBox supports setting a uniform color
            colored_obb = self._create_color_oriented_bbox(bbox,color)
            geo_3dsg.append(colored_obb)
            geo_3dsg.append(bbox)
        
        if display_pc:
            colored_pcds = self._get_colored_pcd(terra)
            all_geoms = geo_3dsg + colored_pcds
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for g in all_geoms:
                vis.add_geometry(g)
            render_opt = vis.get_render_option()
            render_opt.point_size = 2.0  # smaller points
            vis.run()
            vis.destroy_window()
        else:
            o3d.visualization.draw_geometries(geo_3dsg)
    
    def _create_color_oriented_bbox(self, obb, color=(1, 0, 0)):
        """
        Creates a solid mesh for an oriented bounding box.
        """
        full_size = obb.extent
        full_size = np.maximum(full_size, 1e-4)

        # Create a box mesh and transform it
        box_mesh = o3d.geometry.TriangleMesh.create_box(*full_size)
        box_mesh.paint_uniform_color(color)

        # Move to center at origin
        box_mesh.translate(-box_mesh.get_center())

        # Apply rotation and translation
        box_mesh.rotate(obb.R, center=(0, 0, 0))
        box_mesh.translate(obb.center)
        return box_mesh

    def _get_colored_pcd(self, terra):
        count_threshold = 1
        global_pts = {}    
        for idx in range(terra.pc.shape[0]):
            if idx in terra.pcidx_2_clipid.keys():
                max_class, max_count = max(terra.pcidx_2_clipid[idx].items(), key=lambda x: x[1])
                # Make sure max_count is more than some threshold
                if max_count < count_threshold:
                    if -1 in global_pts.keys():
                        global_pts[-1].append(idx)
                    else:
                        global_pts[-1] = [idx]    
                elif max_class in global_pts.keys():
                    global_pts[max_class].append(idx)
                else:
                    global_pts[max_class] = [idx]
            else:
                if -1 in global_pts.keys():
                    global_pts[-1].append(idx)
                else:
                    global_pts[-1] = [idx]
        pcds = []
        for class_id in global_pts.keys():
            pcd = o3d.geometry.PointCloud()
            if class_id != -1:
                if class_id < self.num_terrains:
                    pcd.points = o3d.utility.Vector3dVector(
                        terra.pc[global_pts[class_id], :3])# - np.array([0,0,125]))
                    # pcd.paint_uniform_color(self.terrain_colors[class_id])
                    pcd.paint_uniform_color(self.grays[class_id])
                else:
                    # Otherwise, generate a random color
                    random_col = np.random.rand(3).tolist()
                    pcd.points = o3d.utility.Vector3dVector(
                        terra.pc[global_pts[class_id], :3])# - np.array([0,0,125]))
                    pcd.paint_uniform_color(random_col)
            pcds.append(pcd)
        return pcds

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
    
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(global_pc[:,:3])
    pcd.paint_uniform_color([0.5,0.5,0.5])
    geometries.extend([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geo in geometries:
        vis.add_geometry(geo)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    vis.run()
    vis.destroy_window()
    
    with open(args.terra_3dsg, "rb") as f:
        terra_3dsg = pkl.load(f)
    
    tv = TerraVisualizer(level_offset=50)
    
    # Display Regions 
    tv.display_regions(terra_3dsg)
    
    # Display Places
    tv.display_places(terra_3dsg)
    
    # Display full 3DSG with point cloud
    tv.display_3dsg(terra_3dsg)
    tv.display_3dsg(terra_3dsg,pc=global_pc[:,:3])