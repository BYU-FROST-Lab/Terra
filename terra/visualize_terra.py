from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d

from utils import numeric_key
from terra_utils import copy_obb

class TerraVisualizer():
    def __init__(self, level_offset, terrain_colors=None, num_terrains=None):
        self.level_offset = level_offset
        cmap = plt.get_cmap("tab10")  # 10 distinct colors
        if terrain_colors:
            self.num_terrains = len(terrain_colors)
            self.terrain_colors = terrain_colors
        elif num_terrains:
            self.terrain_colors = [cmap(i % 10)[:3] for i in range(num_terrains)]
        else:
            self.num_terrains = 3
            self.terrain_colors = [cmap(i % 10)[:3] for i in range(self.num_terrains)]
        self.grays = [[0.3,0.3,0.3],[0.7,0.7,0.7],[0.1,0.1,0.1],[0.9,0.9,0.9]]
           
    def display_places(self, G, pc=None):
        places_subgraph = G.subgraph(
            [n_id for n_id in list(G.nodes) if G.nodes[n_id]["level"] == 1]
        )
        self.display_3dsg(places_subgraph, pc=pc)
    
    def display_regions(self, G, pc=None):
        regions_subgraph = G.subgraph(
            [n_id for n_id in list(G.nodes) if G.nodes[n_id]["level"] > 1]
        )
        self.display_3dsg(regions_subgraph, pc=pc)
    
    def display_3dsg(self, 
                     G, 
                     node_colors=None, 
                     pc=None, 
                     plot_objects_on_ground=False, 
                     return_geo=False):
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
            
            if G.nodes[n_id]["level"] == 0 and plot_objects_on_ground:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5).translate([xy[0],xy[1],xy[2]])
            else:
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
            if G.nodes[n_id]["level"] == 0 and plot_objects_on_ground:
                points.append([xy[0], xy[1], 0])
            else:
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

    def display_terra(self, 
                      terra, 
                      display_pc=False, 
                      plot_objects_on_ground=False, 
                      color_pc_clip=True, 
                      color_terrain=False):
        geo_3dsg = self.display_3dsg(terra.terra_3dsg, plot_objects_on_ground=plot_objects_on_ground, return_geo=True)
        
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
            if plot_objects_on_ground:
                bbox.translate([0, 0, 0], relative=True)
            else:
                bbox.translate([0, 0, self.level_offset-z_init], relative=True)
            task_idx = tobj.get_task_idx()
            color = task_colors[task_idx]
            bbox.color = (0,0,0)#color  # OrientedBoundingBox supports setting a uniform color
            colored_obb = self._create_color_oriented_bbox(bbox,color)
            geo_3dsg.append(colored_obb)
            geo_3dsg.append(bbox)
        
        if display_pc:
            colored_pcds = self._get_colored_pcd(terra, color_pc_clip, color_terrain)
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

    def display_task_relevant_places(self, 
                                     terra_3dsg, 
                                     region_tasks, 
                                     task_relevant_place_nodes, 
                                     pc, 
                                     task_idx=-1, 
                                     region_scores=None):
        
        # If region_scores is provided, display heatmap mode
        if region_scores is not None:
            region_nodes = [n for n, d in terra_3dsg.nodes(data=True) if d["level"] > 1]
            region_subgraph = terra_3dsg.subgraph(region_nodes)
            for t_idx, place_nodes in task_relevant_place_nodes.items():
                if task_idx != -1 and t_idx != task_idx:
                    continue
                max_score = region_scores[:, t_idx].max().item()
                min_score = region_scores[:, t_idx].min().item()
                region_colors = {}
                for region_idx in range(region_scores.shape[0]):
                    normalized_score = ((region_scores[region_idx,t_idx] - min_score) / (max_score - min_score)).item()
                    region_colors[region_nodes[region_idx]] = plt.cm.jet(normalized_score)[:3]
                print(f"Displaying heatmap of region nodes for task: {region_tasks[t_idx]}")
                self.display_3dsg(region_subgraph, node_colors=region_colors, pc=pc)      
        
        prev_offset = self.level_offset 
        self.level_offset = 1
        if task_idx == -1:
            print("Displaying relevant places for all region monitoring tasks.")
            print("Tasks:", region_tasks)
            all_relevant_places = []
            for t_idx, place_nodes in task_relevant_place_nodes.items():
                all_relevant_places.extend(list(place_nodes))
            relevant_places_subgraph = terra_3dsg.subgraph(all_relevant_places)
            self.display_3dsg(relevant_places_subgraph, pc=pc)
        else:
            print(f"Displaying relevant places for task: {region_tasks[task_idx]}")
            place_nodes = task_relevant_place_nodes[task_idx]
            relevant_places_subgraph = terra_3dsg.subgraph(place_nodes)
            self.display_3dsg(relevant_places_subgraph, pc=pc)
        self.level_offset = prev_offset
 
    def display_path(self, terra_3dsg, path_node_list, pc):
        prev_offset = self.level_offset 
        self.level_offset = 1
        
        place_nodes = [n for n, d in terra_3dsg.nodes(data=True) if d["level"] == 1]
        place_subgraph = terra_3dsg.subgraph(place_nodes)
        colors = {}
        for n in place_nodes:
            if n == path_node_list[0]:
                colors[n] = (0,0,0)
            elif n == path_node_list[-1]:
                colors[n] = (1,0,0)
            elif n in path_node_list[1:-1]:
                colors[n]= (0,1,1)
            else:
                colors[n] = (0.75,0.75,0.75)    
        self.display_3dsg(place_subgraph, node_colors=colors, pc=pc)#, node_rad=3)
        self.level_offset = prev_offset
    
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

    def _get_colored_pcd(self, terra, color_clip=True, color_terrain=False):
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
                        terra.pc[global_pts[class_id], :])# - np.array([0,0,125]))
                    if color_terrain:
                        pcd.paint_uniform_color(self.terrain_colors[class_id])
                    else:
                        pcd.paint_uniform_color(self.grays[class_id])
                elif color_clip:
                    # Otherwise, generate a random color
                    color = np.random.rand(3).tolist()
                    pcd.points = o3d.utility.Vector3dVector(
                        terra.pc[global_pts[class_id], :])# - np.array([0,0,125]))
                    pcd.paint_uniform_color(color)
                else:
                    continue
                pcds.append(pcd)
        if not color_clip:
            color = [0.5,0.5,0.5]
            points = np.concatenate(
                [terra.pc[v,:] for k, v in global_pts.items() if k >= self.num_terrains],
                axis=0
            )
            pcd.points = o3d.utility.Vector3dVector(points)# - np.array([0,0,125]))
            pcd.paint_uniform_color(color)
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
    parser.add_argument('--num_terrains',
                        type=int,
                        default=3,
                        help="Number of terrain classes YOLO model used")
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
    
    tv = TerraVisualizer(level_offset=50, num_terrains=args.num_terrains)
    
    # Display Regions 
    tv.display_regions(terra_3dsg)
    
    # Display Places
    tv.display_places(terra_3dsg)
    
    # Display full 3DSG with point cloud
    tv.display_3dsg(terra_3dsg)
    tv.display_3dsg(terra_3dsg,pc=global_pc[:,:3])