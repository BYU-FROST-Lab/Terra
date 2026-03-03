from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as mp_Path

import heapq

import open3d as o3d

from terra.utils import numeric_key
from terra.utils import copy_obb


def generate_grays(n, avoid=0.5, eps=1e-9):
    result = []
    
    # Start with two intervals split around the avoided value
    heap = [
        (-(avoid - 0.0), 0.0, avoid),
        (-(1.0 - avoid), avoid, 1.0)
    ]
    
    while len(result) < n:
        length, a, b = heapq.heappop(heap)
        mid = (a + b) / 2
        
        # Skip if too close to avoided value
        if abs(mid - avoid) < eps:
            continue
            
        result.append([mid, mid, mid])
        
        heapq.heappush(heap, (-(mid - a), a, mid))
        heapq.heappush(heap, (-(b - mid), mid, b))
        
    return result


class TerraVisualizer():
    def __init__(self, level_offset, terrain_colors=None, num_terrains=None):
        self.level_offset = level_offset
        cmap = plt.get_cmap("tab10")  # 10 distinct colors
        if terrain_colors:
            self.num_terrains = len(terrain_colors)
            self.terrain_colors = terrain_colors
        elif num_terrains:
            self.num_terrains = num_terrains
            self.terrain_colors = [cmap(i % 10)[:3] for i in range(num_terrains)]
        else:
            self.num_terrains = 3
            self.terrain_colors = [cmap(i % 10)[:3] for i in range(self.num_terrains)]
        self.grays = generate_grays(self.num_terrains)
           
    def display_places(self, G, pc=None, plot_ids=False, no_spheres=False, return_geo=False):
        places_subgraph = G.subgraph(
            [n_id for n_id in list(G.nodes) if G.nodes[n_id]["level"] == 1]
        )
        if no_spheres:
            return self.display_3dsg_points(places_subgraph, pc=pc, plot_ids=plot_ids, return_geo=return_geo)
        else:
            return self.display_3dsg(places_subgraph, pc=pc, plot_ids=plot_ids, return_geo=return_geo)
   
    def display_regions(self, G, pc=None, plot_ids=False, return_geo=False):
        regions_subgraph = G.subgraph(
            [n_id for n_id in list(G.nodes) if G.nodes[n_id]["level"] > 1]
        )
        return self.display_3dsg(regions_subgraph, pc=pc, plot_ids=plot_ids, return_geo=return_geo)
    
    def display_3dsg(self, 
                     G, 
                     node_colors=None, 
                     pc=None, 
                     plot_objects_on_ground=False, 
                     return_geo=False,
                     plot_ids=False):
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

            if plot_ids:
                text_label = o3d.t.geometry.TriangleMesh.create_text(str(n_id), depth=1.2).to_legacy()
                text_label.paint_uniform_color([0, 0, 0])  # or any color
                text_label.transform([[0.05, 0, 0, xy[0]], [0, 0.05, 0, xy[1]], [0, 0, 0.1, z+2],
                                            [0, 0, 0, 1]])
                nodes.append(text_label)
            
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
        
        if len(colors) > 0:
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

    def display_3dsg_points(self, 
                         G, 
                         node_colors=None, 
                         pc=None, 
                         plot_objects_on_ground=False, 
                         return_geo=False,
                         plot_ids=False):
        geometries = []

        node_points = []
        node_colors_arr = []
        node_idx_map = {}

        text_geometries = []

        # ---- Nodes (POINTS instead of spheres) ----
        for i, n_id in enumerate(G.nodes):
            level_num = G.nodes[n_id]["level"] + 1
            z = level_num * self.level_offset
            xy = G.nodes[n_id]["pos"]

            if G.nodes[n_id]["level"] == 0 and plot_objects_on_ground:
                p = [xy[0], xy[1], 0]
            else:
                p = [xy[0], xy[1], z]

            node_points.append(p)
            node_idx_map[n_id] = i

            # ---- Color logic (unchanged) ----
            if node_colors is not None:
                if n_id in node_colors:
                    node_colors_arr.append(node_colors[n_id])
                else:
                    node_colors_arr.append([0, 0, 0])
            elif G.nodes[n_id]["terrain_id"] > -1:
                node_colors_arr.append(
                    self.terrain_colors[G.nodes[n_id]["terrain_id"]]
                )
            else:
                node_colors_arr.append([0, 0, 0])

            # ---- Optional text labels ----
            if plot_ids:
                text_label = o3d.t.geometry.TriangleMesh.create_text(
                    str(n_id), depth=1.2
                ).to_legacy()
                text_label.paint_uniform_color([0, 0, 0])
                text_label.transform([
                    [0.05, 0,    0,    p[0]],
                    [0,    0.05, 0,    p[1]],
                    [0,    0,    0.1,  p[2] + 2],
                    [0,    0,    0,    1]
                ])
                text_geometries.append(text_label)

        # ---- Create single PointCloud for all nodes ----
        node_pcd = o3d.geometry.PointCloud()
        node_pcd.points = o3d.utility.Vector3dVector(np.asarray(node_points))
        node_pcd.colors = o3d.utility.Vector3dVector(np.asarray(node_colors_arr))

        geometries.append(node_pcd)
        geometries.extend(text_geometries)

        # ---- Edges (LineSet) ----
        lines = []
        line_colors = []

        for (u, v) in G.edges:
            if u in node_idx_map and v in node_idx_map:
                lines.append([node_idx_map[u], node_idx_map[v]])
                line_colors.append([0.75, 0.75, 0.75])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(node_points))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors))

        geometries.append(line_set)

        # ---- Optional external point cloud ----
        if pc is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(pcd)

        # ---- Return or visualize ----
        if return_geo:
            return geometries
        else:
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            for geo in geometries:
                vis.add_geometry(geo)

            render_opt = vis.get_render_option()
            render_opt.point_size = 6.0  # make nodes visible without spheres

            vis.run()
            vis.destroy_window()

    def display_point_cloud(self, terra, pc, ms_map=False,point_size=2.0, color=[0.5, 0.5, 0.5]):
        """
        Display only a point cloud.

        Args:
            pc (np.ndarray): Nx3 or Nx4 array of points.
            point_size (float): Size of rendered points.
            color (list): RGB color for the point cloud (ignored if pc has per-point colors).
        """
        if pc is None:
            raise ValueError("Point cloud (pc) cannot be None.")

        # If pc has more than 3 columns (e.g., XYZ + intensity), keep only XYZ
        if pc.shape[1] > 3:
            pc = pc[:, :3]

        if ms_map:
            colored_pcds = self._get_colored_pcd(terra, color_clip=True, color_terrain=True)
            all_geoms = colored_pcds
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for g in all_geoms:
                vis.add_geometry(g)
            render_opt = vis.get_render_option()
            render_opt.point_size = 2.0  # smaller points
            vis.run()
            vis.destroy_window()

        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)

            # If no per-point colors exist, paint uniformly
            pcd.paint_uniform_color(color)

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)

            render_opt = vis.get_render_option()
            render_opt.point_size = point_size

            vis.run()
            vis.destroy_window()     

    def display_selected_nodes(self, G, selected_node_ids, pc=None):
        selected_subgraph = G.subgraph(selected_node_ids)
        self.display_3dsg(selected_subgraph, pc=pc)

    def get_nodes_in_rectangle_from_refs(self, G, ref_node_ids):
        """
        ref_node_ids: iterable of 4 node IDs defining the rectangle extremes
        Returns: list of node IDs inside the rectangle (inclusive)
        """

        if len(ref_node_ids) != 4:
            raise ValueError("Exactly 4 reference node IDs are required")

        # Collect XY positions of the reference nodes
        ref_points = []
        for n_id in ref_node_ids:
            if n_id not in G.nodes:
                raise KeyError(f"Node ID {n_id} not found in graph")
            x, y = G.nodes[n_id]["pos"][:2]
            ref_points.append([x, y])

        ref_points = np.asarray(ref_points)

        # Rectangle bounds
        min_x, min_y = np.min(ref_points, axis=0)
        max_x, max_y = np.max(ref_points, axis=0)

        # Select nodes inside rectangle
        selected_ids = []
        for n_id in G.nodes:
            x, y = G.nodes[n_id]["pos"][:2]
            level = G.nodes[n_id]["level"]

            if min_x <= x <= max_x and min_y <= y <= max_y and level == 1:  # Only consider place nodes
                selected_ids.append(n_id)


        return selected_ids

    def get_nodes_in_diagonal_rectangle(self, G, ref_node_ids):
        """
        ref_node_ids: two nodes defining the diagonal corners of the rectangle
        Returns: list of node IDs inside the rectangle (inclusive)
        """

        if len(ref_node_ids) != 2:
            raise ValueError("Exactly 2 reference node IDs are required")

        # Collect XY positions of the reference nodes
        ref_points = []
        for n_id in ref_node_ids:
            if n_id not in G.nodes:
                raise KeyError(f"Node ID {n_id} not found in graph")
            x, y = G.nodes[n_id]["pos"][:2]
            ref_points.append([x, y])

        ref_points = np.asarray(ref_points)

        # Rectangle bounds
        min_x, min_y = np.min(ref_points, axis=0)
        max_x, max_y = np.max(ref_points, axis=0)

        # Select nodes inside rectangle
        selected_ids = []
        for n_id in G.nodes:
            x, y = G.nodes[n_id]["pos"][:2]
            level = G.nodes[n_id]["level"]

            if min_x <= x <= max_x and min_y <= y <= max_y and level == 1:  # Only consider place nodes
                selected_ids.append(n_id)

        return selected_ids

    def get_nodes_in_polygon_from_sides(self, G, side_node_pairs):
        """
        side_node_pairs: list of 4 tuples [(id1, id2), ...]
                        each tuple defines one polygon side

        Returns: list of node IDs inside the polygon
        """
        # Convert 4 sides (8 nodes) into polygon vertices
        polygon = self.build_polygon_from_8_indices(
            G, [n for pair in side_node_pairs for n in pair]
        )

        # Create a Path object for point-in-polygon test
        poly_path = mp_Path(polygon)

        selected_ids = []

        # Check each node
        for n_id in G.nodes:
            xy = G.nodes[n_id]["pos"][:2]  # take XY only
            level = G.nodes[n_id]["level"]
            if level != 1: # Only consider place nodes
                continue
            if poly_path.contains_point(xy) or n_id in [n for pair in side_node_pairs for n in pair]: # Include boundary nodes
                selected_ids.append(n_id)
        

        # Display polygon and selected nodes (for debugging)
        plt.figure()
        plt.plot(*polygon[[0,1,2,3,0]].T, 'r-')  # polygon outline
        selected_points = np.array([G.nodes[n_id]["pos"][:2] for n_id in selected_ids])
        plt.scatter(selected_points[:,0], selected_points[:,1], c='b')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Selected Nodes Inside Polygon")
        plt.axis('equal')
        plt.show()


        return selected_ids

    def build_polygon_from_8_indices(self, G, indices):
        """
        indices: list of 8 node IDs, two per side (side0, side1, side2, side3)
        Returns: 4 polygon vertices in order (numpy array Nx2)
        """
        if len(indices) != 8:
            raise ValueError("Exactly 8 node IDs required")

        # Get XY positions of the 8 nodes
        points = [G.nodes[n_id]["pos"][:2] for n_id in indices]

        # Each side: two points
        side0 = points[0:2]
        side1 = points[2:4]
        side2 = points[4:6]
        side3 = points[6:8]

        # Polygon vertices: intersections of adjacent sides
        # Order: top-left, top-right, bottom-right, bottom-left (or similar)
        v0 = self._line_intersection(side0[0], side0[1], side3[0], side3[1])
        v1 = self._line_intersection(side0[0], side0[1], side1[0], side1[1])
        v2 = self._line_intersection(side1[0], side1[1], side2[0], side2[1])
        v3 = self._line_intersection(side2[0], side2[1], side3[0], side3[1])

        polygon = np.array([v0, v1, v2, v3])
        return polygon

    def _line_intersection(self, p1, p2, p3, p4):
        """
        Find intersection of line (p1,p2) and (p3,p4) in 2D.
        Returns intersection point as (x, y)
        """
        # Convert to numpy arrays
        p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])

        # Line vectors
        A = p2 - p1
        B = p4 - p3
        C = p3 - p1

        # Solve t: intersection = p1 + t*A
        denom = A[0]*B[1] - A[1]*B[0]
        if np.isclose(denom, 0):
            # Lines are parallel; return midpoint as fallback
            return (p1 + p2) / 2
        t = (C[0]*B[1] - C[1]*B[0]) / denom
        intersection = p1 + t*A
        return intersection

    def display_terra(self, 
                      terra, 
                      display_pc=False, 
                      plot_objects_on_ground=False, 
                      color_pc_clip=True, 
                      color_terrain=False,
                      plot_ids=False):
        geo_3dsg = self.display_3dsg(terra.terra_3dsg, plot_objects_on_ground=plot_objects_on_ground, return_geo=True, plot_ids=plot_ids)
        
        # add in bounding boxes for nodes
        num_tasks = max(set(t.get_task_idx() for t in terra.objects)) + 1
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
        self.grays = generate_grays(len(self.terrain_colors))
        global_pts = {}    
        for idx in range(terra.pc.shape[0]):
            if idx in terra.gidx_2_clipcounts.keys():
                max_class, max_count = max(terra.gidx_2_clipcounts[idx].items(), key=lambda x: x[1])
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
                if class_id < len(self.terrain_colors):#self.num_terrains:
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
                [terra.pc[v,:] for k, v in global_pts.items() if k >= len(self.terrain_colors)],#self.num_terrains],
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
    parser.add_argument('--view_json',
                        type=str,
                        help="Filepath to view.json file saved")
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
    tv.display_regions(terra_3dsg, plot_ids=True)
    
    # Display Places
    # tv.display_places(terra_3dsg)
    
    # Display full 3DSG with point cloud
    geo = tv.display_3dsg(terra_3dsg, return_geo=True)
    # tv.display_3dsg(terra_3dsg,pc=global_pc[:,:3])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geo:
        vis.add_geometry(g)
    if args.view_json:
        params = o3d.io.read_pinhole_camera_parameters(args.view_json)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(params)
    vis.run()