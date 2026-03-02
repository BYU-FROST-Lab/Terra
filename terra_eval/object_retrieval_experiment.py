from argparse import ArgumentParser
import glob
import yaml
import re
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import clip

from terra.utils import load_terra

class ObjectEvaluator():
    def __init__(self, num_imgs, terra, **kwargs):
        self.display_num_imgs = num_imgs
        self.terra = terra
        self.data_folder = kwargs['data_folder']
        self.num_cams = kwargs['num_cams']
        
        self.obj_tasks = kwargs['object_tasks']
        self.num_tasks = len(self.obj_tasks)
        self.task_names = [self.obj_tasks[i]["task"] for i in range(self.num_tasks)]
        
        # Camera intrinsics
        self.K_list = [np.array(kwargs[f'cam{nc+1}_K']).reshape(3,3) for nc in range(self.num_cams)]
        self.dist = [np.array(kwargs[f'cam{nc+1}_dist']) for nc in range(self.num_cams)]
        self.IMG_W = int(kwargs['IMG_W'])
        self.IMG_H = int(kwargs['IMG_H'])
        self.newK_list = []
        self.roi = []
        for i in range(self.num_cams):
            newK, roi = cv2.getOptimalNewCameraMatrix(self.K_list[i], self.dist[i], (self.IMG_W, self.IMG_H), 1, (self.IMG_W, self.IMG_H))
            self.newK_list.append(newK)
            self.roi.append(roi)
        
        # Image and transform folders
        self.cam_image_folders = [os.path.join(self.data_folder, f"camera{i}_images") for i in range(1,self.num_cams+1)]
        lidar2cam_transform_folders = [os.path.join(self.data_folder, f"transformations_lidar2cam{i}") for i in range(1,self.num_cams+1)]
        lidar2global_folder = os.path.join(self.data_folder, "transformations_lidar2global")

        # Preload transforms
        self.lidar2cam_transforms = [self.load_lidar2cam_transforms(f, i) for i,f in enumerate(lidar2cam_transform_folders)]
        self.lidar2global_transforms = self.load_lidar2global_transforms(lidar2global_folder)

        # KDTree for place nodes
        self.place_nodes = [n for n, d in terra.terra_3dsg.nodes(data=True) if d["level"] == 1]
        place_pos = np.array([terra.terra_3dsg.nodes[n]["pos"] for n in self.place_nodes])
        self.kdt_places = KDTree(place_pos)
        
        ## Metrics
        self.gt_obj_count = [self.obj_tasks[i]["ground_truth"] for i in range(self.num_tasks)] # N_t
        self.pred_matches = [0 for _ in range(self.num_tasks)] # M_t
        self.task_rel_count = [0 for _ in range(self.num_tasks)] # \bar{N}_t
        self.task_rel_matches = [0 for _ in range(self.num_tasks)] # \bar{M}_t
    
    def evaluate(self):
        self.record_recall_metrics()
        self.record_precision_metrics()
        
        # Print out metrics
        recall = sum(self.pred_matches) / sum(self.gt_obj_count) if sum(self.gt_obj_count) > 0 else 0.0
        precision = sum(self.task_rel_matches) / sum(self.task_rel_count) if sum(self.task_rel_count) > 0 else 0.0
        F1 = 0.0 if (recall + precision) == 0 else 2 * (precision * recall) / (precision + recall)
        
        print(f"\n\nOBJECT METRICS FOR DATASET: {self.data_folder}")
        print(f"Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {F1:.3f}\n\n")
    
    def record_precision_metrics(self):
        print("\n\nCalculating Precision Metrics:")
        task_relevant_objs = self.get_top90percent_objs()
        for t in range(self.num_tasks):
            Nbar_t = len(task_relevant_objs[t])
            self.task_rel_count[t] = Nbar_t
            Mbar_t = self.count_matches(task_relevant_objs[t])
            self.task_rel_matches[t] = Mbar_t            
            print(f"Task {self.task_names[t]}: GT Count = {Nbar_t}, GT Matches = {Mbar_t}")
            
    def record_recall_metrics(self):
        print("\n\nCalculating Recall Metrics:")
        for t in range(self.num_tasks):
            Nt = self.gt_obj_count[t]
            topk_objs = self.get_topk_objs_for_task(t, Nt)
            Mt = self.count_matches(topk_objs, task_idx=t)
            self.pred_matches[t] = Mt
            print(f"Task {self.task_names[t]}: GT Count = {Nt}, GT Matches = {Mt}")
    
    def get_top90percent_objs(self):
        max_task_scores = {task_idx: 0.0 for task_idx in range(self.num_tasks)}
        pred_task_objs = {task_idx: [] for task_idx in range(self.num_tasks)}
        pred_objects = self.terra.objects
        for pred_obj in pred_objects:
            task_idx = pred_obj.get_task_idx()
            score = pred_obj.get_top_score()
            pred_task_objs[task_idx].append(pred_obj)
            if score > max_task_scores[task_idx]:
                max_task_scores[task_idx] = score
        
        top90perc_objs = {task_idx: [] for task_idx in range(self.num_tasks)}
        for t in range(self.num_tasks):
            for obj in pred_task_objs[t]:
                score = obj.get_top_score()
                if score > (0.9 * max_task_scores[t]):
                    top90perc_objs[t].append(obj)
        
        return top90perc_objs
        
    def get_topk_objs_for_task(self, task_idx, k):
        pred_objects = self.terra.objects
        # Sort objects by their score for the given task index, descending
        sorted_objs = sorted(
            pred_objects,
            key=lambda obj: obj.get_task_scores()[task_idx].item(),
            reverse=True
        )
        # Pick the top-k
        topk_objs = sorted_objs[:k]
        return topk_objs

    def count_matches(self, objects, task_idx=None):
        num_matches = 0
        print(f"Count matches out of {len(objects)} objects")
        
        for obj_idx, tobj in enumerate(objects):
            obb = tobj.get_bbox()
            if task_idx is None:
                task_idx = tobj.get_task_idx()
            obb_corners = np.asarray(obb.get_box_points())
            edges = self.build_obb_edges(obb_corners)
            obb_center_xy = obb.center[:2]

            # Nearest place node
            _, idx = self.kdt_places.query(obb_center_xy)
            closest_place_node = self.place_nodes[idx]
            
            # # ####
            # # # DEBUGGING
            # ## DISPLAY PCD + Place Nodes
            # place_nodes = [n for n, d in self.terra.terra_3dsg.nodes(data=True) if d["level"] == 1]
            # place_subgraph = self.terra.terra_3dsg.subgraph(place_nodes)
            # colors = {}
            # for n in place_nodes:
            #     if n == closest_place_node:
            #         colors[n] = (0,0,1.0)
            #     else:
            #         colors[n] = (0.75,0.75,0.75)    
            # self.terra.visualizer.level_offset = 0.0 
            # geo = self.terra.visualizer.display_3dsg(place_subgraph, node_colors=colors, pc=self.terra.pc, return_geo=True, plot_ids=True)
            # self.terra.visualizer.level_offset = 50.0
            
            # ## OR just DISPLAY PCD
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(self.terra.pc)
            # pcd.paint_uniform_color([0.5,0.5,0.5])
            
            # spheres = []
            # for corner in obb_corners:
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5).translate([corner[0],corner[1],corner[2]])
            #     sphere.paint_uniform_color([1.0, 0.0, 0.0])
            #     spheres.append(sphere)
            # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5).translate([obb.center[0],obb.center[1],obb.center[2]])
            # sphere.paint_uniform_color([0.0, 1.0, 0.0])
            # spheres.append(sphere)
            # o3d.visualization.draw_geometries(geo + spheres)
            # # # o3d.visualization.draw_geometries([pcd] + spheres)
            # # # ####

            # Images that see this node
            img_indices = self.terra.nodeid_2_img_idx[closest_place_node]

            shown = 0
            img_buffer = []
            
            for img_idx in img_indices:
                if self.display_num_imgs != -1 and shown >= self.display_num_imgs:
                    break

                img_path = self.terra.img_names[img_idx]
                cam_id, timestamp = self.parse_camera_and_timestamp(img_path)

                img_full_path = os.path.join(
                    self.cam_image_folders[cam_id],
                    os.path.basename(img_path)
                )
                if not os.path.exists(img_full_path):
                    continue

                dist_img = cv2.imread(img_full_path)
                dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)

                # Load closest transforms
                lidar2cam_file = self.find_closest_transform(timestamp, self.lidar2cam_transforms[cam_id])
                lidar2global_file = self.find_closest_transform(timestamp, self.lidar2global_transforms)

                T_lidar2cam = self.load_transformation(lidar2cam_file)
                T_lidar2global = self.load_transformation(lidar2global_file)
                
                # ############# DISPLAY 3D Camera COORD ###################
                # print("Camera Image:",img_full_path)
                # pc_h = np.hstack([self.terra.pc[:,:3], np.ones((self.terra.pc.shape[0],1))]) # (N,4)
                # # for nc in range(self.num_cams):
                # #     lidar2cam_file = self.find_closest_transform(timestamp, self.lidar2cam_transforms[nc])
                # #     T_lidar2cam = self.load_transformation(lidar2cam_file)
                # print("Lidar 2 cam:",T_lidar2cam)
                # # cam_points_h = pc_h @ (T_lidar2cam @ np.linalg.inv(T_lidar2global)).T # (N,4)
                # T_lidar2global_inv = np.eye(4)
                # T_lidar2global_inv[:3,:3] = T_lidar2global[:3,:3].T
                # T_lidar2global_inv[:3,3] = - T_lidar2global[:3,:3].T @ T_lidar2global[:3,3]
                # cam_points_h = pc_h @ (T_lidar2cam @ T_lidar2global_inv).T # (N,4)
                # cam_points = cam_points_h[:,:3] # (N,3)
                
                # proj_pts = cam_points @ self.newK_list[cam_id].T
                # # Keep points in front of camera
                # zs = proj_pts[:, 2]
                # valid_z = zs > 0
                # xs = np.round(proj_pts[:, 0] / zs).astype(int)
                # ys = np.round(proj_pts[:, 1] / zs).astype(int)
                
                # in_bounds = (
                #     valid_z &
                #     (xs >= self.roi[cam_id][0]) & 
                #     (xs < (self.roi[cam_id][0]+self.roi[cam_id][2])) & 
                #     (ys >= self.roi[cam_id][1]) & 
                #     (ys < (self.roi[cam_id][1]+self.roi[cam_id][3]))
                # )
                # xs, ys = xs[in_bounds], ys[in_bounds]
                # cam_points = cam_points[in_bounds]
                
                # colors = np.full((self.terra.pc.shape[0], 3), [0.5, 0.5, 0.5])  # gray
                # colors[in_bounds] = [1.0, 0.0, 1.0]        # red for FOV points

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(self.terra.pc)
                # # pcd.paint_uniform_color([0.5,0.5,0.5])
                # pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # spheres = []
                # for corner in obb_corners:
                #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5).translate([corner[0],corner[1],corner[2]])
                #     sphere.paint_uniform_color([1.0, 0.0, 0.0])
                #     spheres.append(sphere)
                # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5).translate([obb.center[0],obb.center[1],obb.center[2]])
                # sphere.paint_uniform_color([0.0, 1.0, 0.0])
                # spheres.append(sphere)
                # # Math for camera location
                # cam_origin_h_in_Cframe = np.array([0,0,0,1])
                # cam_origin_h_in_Gframe = T_lidar2global @ np.linalg.inv(T_lidar2cam) @ cam_origin_h_in_Cframe
                # cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                # cam_sphere.translate([cam_origin_h_in_Gframe[0],cam_origin_h_in_Gframe[1],cam_origin_h_in_Gframe[2]])
                # cam_sphere.paint_uniform_color([0.0, 0.0, 1.0])
                # spheres.append(cam_sphere)
                # # Math for lidar originW
                # lidar_origin_h_in_Lframe = np.array([0,0,0,1])
                # lidar_origin_h_in_Gframe = T_lidar2global @ lidar_origin_h_in_Lframe
                # lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                # lidar_sphere.translate([lidar_origin_h_in_Gframe[0],lidar_origin_h_in_Gframe[1],lidar_origin_h_in_Gframe[2]])
                # lidar_sphere.paint_uniform_color([0.0, 0.0, 0.0])
                # spheres.append(lidar_sphere)
                
                # lidar_x_origin_h_in_Lframe = np.array([0.5,0,0,1])
                # lidar_x_origin_h_in_Gframe = T_lidar2global @ lidar_x_origin_h_in_Lframe
                # lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                # lidar_sphere.translate([lidar_x_origin_h_in_Gframe[0],lidar_x_origin_h_in_Gframe[1],lidar_x_origin_h_in_Gframe[2]])
                # lidar_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                # spheres.append(lidar_sphere)
                
                # lidar_y_origin_h_in_Lframe = np.array([0,0.5,0,1])
                # lidar_y_origin_h_in_Gframe = T_lidar2global @ lidar_y_origin_h_in_Lframe
                # lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                # lidar_sphere.translate([lidar_y_origin_h_in_Gframe[0],lidar_y_origin_h_in_Gframe[1],lidar_y_origin_h_in_Gframe[2]])
                # lidar_sphere.paint_uniform_color([0.0, 1.0, 0.0])
                # spheres.append(lidar_sphere)
                
                # lidar_z_origin_h_in_Lframe = np.array([0,0,0.5,1])
                # lidar_z_origin_h_in_Gframe = T_lidar2global @ lidar_z_origin_h_in_Lframe
                # lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                # lidar_sphere.translate([lidar_z_origin_h_in_Gframe[0],lidar_z_origin_h_in_Gframe[1],lidar_z_origin_h_in_Gframe[2]])
                # lidar_sphere.paint_uniform_color([0.0, 0.0, 1.0])
                # spheres.append(lidar_sphere)
                
                
                
                # # o3d.visualization.draw_geometries(geo + spheres)
                # o3d.visualization.draw_geometries([pcd] + spheres)
                # # ###########################################################

                # Project 3D bbox
                bbox_2d, valid = self.project_box_3dto2d(
                    obb_corners, T_lidar2global, T_lidar2cam, self.newK_list[cam_id]
                ) # [8x2]
                
                center_2d = self.project_point_3dto2d(
                    obb.center, T_lidar2global, T_lidar2cam, self.newK_list[cam_id], self.IMG_H, self.IMG_W
                    #img.shape[0], img.shape[1]
                )
                if center_2d is None:
                    continue
                
                ## Draw on image
                img = cv2.undistort(dist_img, self.K_list[cam_id], self.dist[cam_id], None, self.newK_list[cam_id])
                img = self.draw_center_point(img, center_2d)
                img = self.draw_projected_obb(img, bbox_2d, edges, valid)
                img = self.draw_task(img, self.task_names[task_idx], position=(10, 30))
                
                # Create figure but DO NOT show yet
                # plt.figure(figsize=(8, 6))
                # plt.imshow(img)
                # plt.title(
                #     f"Object {obj_idx} | {os.path.basename(img_path)} | place {closest_place_node}"
                # )
                # plt.axis("off")
                title = f"{os.path.basename(img_path)} | place {closest_place_node}"
                img_buffer.append((img, title))

                shown += 1
                
                if len(img_buffer) == 9:
                    self._show_image_batch(
                        img_buffer, obj_idx, task_idx
                    )
                    img_buffer.clear()
            
            if img_buffer:
                self._show_image_batch(
                    img_buffer, obj_idx, task_idx
                )

            if shown > 0:
                decision = self._wait_for_yn()
                
                if decision is None:
                    assert False, f"[Object {obj_idx}] for task {self.task_names[task_idx]}. No decision made!"
                elif decision:
                    print(f"[Object {obj_idx}] for task {self.task_names[task_idx]}. Matched (y)")
                    num_matches += 1
                else:
                    print(f"[Object {obj_idx}] for task {self.task_names[task_idx]}. Not a match or GT already detected (n)")
            else:
                print(f"\n[Object {obj_idx}] for task {self.task_names[task_idx]}. No images see bounding box centroid!\n")
        return num_matches
    
    
    def _wait_for_yn(self):
        """
        Blocks until user presses 'y' or 'n' in the matplotlib window.
        Returns True for 'y', False for 'n'.
        """
        decision = {"value": None}

        def on_key(event):
            if event.key in ("y", "n"):
                decision["value"] = (event.key == "y")
                for fig_num in plt.get_fignums():
                    plt.close(fig_num)

        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()  # blocks
        # fig.canvas.mpl_disconnect(cid)

        return decision["value"]
    
    def _show_image_batch(self, img_buffer, obj_idx, task_idx):
        """
        Display up to 6 images in a 3x3 subplot figure.
        """
        fig, axes = plt.subplots(3, 3, figsize=(10, 15))
        axes = axes.flatten()

        for ax, (img, title) in zip(axes, img_buffer):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        # Hide unused subplots
        for ax in axes[len(img_buffer):]:
            ax.axis("off")

        fig.suptitle(
            f"Object {obj_idx} | Task: {self.task_names[task_idx]}\nPress Y / N",
            fontsize=14
        )

        fig.tight_layout(rect=[0, 0, 1, 1])

    
    def build_obb_edges(self, corners_3d, angle_eps=1e-3):
        edges = set()
        for i in range(8):
            p = corners_3d[i]

            # distances to all other points
            dists = []
            for j in range(8):
                if i == j:
                    continue
                v = corners_3d[j] - p
                dists.append((j, np.linalg.norm(v), v))

            # sort by distance
            dists.sort(key=lambda x: x[1])

            # smallest two are always connected
            j1, _, v1 = dists[0]
            j2, _, v2 = dists[1]
            edges.add(frozenset((i, j1)))
            edges.add(frozenset((i, j2)))

            # determine 3rd edge via perpendicularity
            for j, _, v in dists[2:4]:  # only 3rd or 4th can be correct
                if (abs(np.dot(v, v1)) < angle_eps and
                    abs(np.dot(v, v2)) < angle_eps):
                    edges.add(frozenset((i, j)))
                    break
        return edges

    # ---------------------------
    # Transformation helpers
    # ---------------------------
    def load_transformation(self, file_path):
        transform_1d = np.load(file_path)
        trans_mat = np.eye(4)
        rot = R.from_quat(transform_1d[3:])
        trans_mat[:3, :3] = rot.as_matrix()
        trans_mat[:3, 3] = transform_1d[:3]
        return trans_mat

    def project_point_3dto2d(self, pt, T_lidar2global, T_lidar2cam, K, h, w):
        pt_h = np.array([pt[0], pt[1], pt[2], 1.0])  # (4,)
        cam_pt = (T_lidar2cam @ np.linalg.inv(T_lidar2global) @ pt_h)[:3]

        proj = K @ cam_pt
        z = proj[2]
        if z <= 1e-6:
            return None  # behind camera
        x = int(proj[0] / z)
        y = int(proj[1] / z)

        if x < 0 or x >= w or y < 0 or y >= h:
            return None 
        
        return np.array([x, y])

    def project_box_3dto2d(self, bbox_3d, T_lidar2global, T_lidar2cam, K):
        corners_h = np.hstack([bbox_3d, np.ones((bbox_3d.shape[0],1))]) # (8,4)
        cam_points_h = corners_h @ (T_lidar2cam @ np.linalg.inv(T_lidar2global)).T # (8,4)
        cam_points = cam_points_h[:,:3] # [8x3]
        
        proj_pts = cam_points @ K.T
        # Keep points in front of camera
        zs = proj_pts[:, 2]
        valid = zs > 0
        xs = (proj_pts[:,0] / zs).astype(int)
        ys = (proj_pts[:,1] / zs).astype(int)
        
        return np.vstack([xs, ys]).T, valid

    # ---------------------------
    # Drawing helpers
    # ---------------------------
    def draw_projected_obb(self, img, bbox_2d, edges, valid, color=(255, 0, 0), thickness=2):
        h, w = img.shape[:2]
        for edge in edges:
            i, j = tuple(edge)
            if not (valid[i] and valid[j]):
                continue   
            p1 = np.round(bbox_2d[i]).astype(int)
            p2 = np.round(bbox_2d[j]).astype(int)
            
            p1[0] = np.clip(p1[0], 0, w-1)
            p1[1] = np.clip(p1[1], 0, h-1)
            p2[0] = np.clip(p2[0], 0, w-1)
            p2[1] = np.clip(p2[1], 0, h-1)

            cv2.line(img, tuple(p1), tuple(p2), color, thickness)
        return img
    
    def draw_center_point(self, img, pt_2d, color=(0, 255, 0), radius=5):
        if pt_2d is None:
            return img
        h, w = img.shape[:2]
        x, y = pt_2d
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1)  # filled circle
        return img
    
    def draw_task(self, img, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale=0.8, font_color=(0,0,0), bg_color=(255,255,255),
                                thickness=2, padding=5):
        x, y = position
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Draw rectangle
        cv2.rectangle(img, (x - padding, y - text_h - padding), 
                            (x + text_w + padding, y + baseline + padding), 
                            bg_color, -1)
        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        return img

    # ---------------------------
    # Filename parsing
    # ---------------------------
    def parse_camera_and_timestamp(self, img_path):
        img_name = os.path.basename(img_path)
        m = re.match(r"cam(\d+)_img_([0-9]+\.[0-9]+)", img_name)
        if m is None:
            raise ValueError(f"Cannot parse {img_name}")
        cam_id = int(m.group(1)) - 1  # 0-indexed
        timestamp = float(m.group(2))
        return cam_id, timestamp

    def load_camera_image_timestamps(self, cam_folder, cam_id):
        img_paths = sorted(glob.glob(f"{cam_folder}/cam{cam_id+1}_img_*.jpg"))
        return [(float(os.path.basename(p).split("_")[-1].replace(".jpg","")), p) for p in img_paths]

    def load_lidar2cam_transforms(self, transform_folder, cam_id):
        files = sorted(glob.glob(f"{transform_folder}/transform_lidar_to_cam{cam_id+1}_*.npy"))
        return [(float(os.path.basename(f).split("_")[-1].replace(".npy","")), f) for f in files]

    def load_lidar2global_transforms(self, transform_folder):
        files = sorted(glob.glob(f"{transform_folder}/transform_lidar_to_map_*.npy"))
        return [(float(os.path.basename(f).split("_")[-1].replace(".npy","")), f) for f in files]

    def find_closest_transform(self, cam_ts, transforms_list):
        times = [t[0] for t in transforms_list]
        idx = min(range(len(times)), key=lambda i: abs(times[i]-cam_ts))
        return transforms_list[idx][1]


# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, help="YAML file with object tasks and Terra path")
    parser.add_argument('--num_imgs', type=int, default=-1, help="Number of images to process (-1 = all)")
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        cfg = yaml.safe_load(f)
    print("Number of cameras:", cfg['num_cams'])
    
    # Load Terra
    terra = load_terra(cfg['terra'])
    terra.alpha = cfg['alpha']
    print("Alpha parameter for object prediction:", terra.alpha)

    # Setup CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/16", device=device)

    # Encode tasks
    tasks = [task["task"] for task in cfg['object_tasks']]
    tasks[:0] = terra.terrain_names
    input_task_embs = [clip_model.encode_text(clip.tokenize([t]).to(device)).float() for t in tasks]
    input_task_tensor = torch.vstack(input_task_embs)
    input_task_tensor.div_(input_task_tensor.norm(dim=-1, keepdim=True))

    # Predict objects
    terra.predict_objects(input_task_tensor, tasks[terra.num_terrain:], cfg['prediction_method'])
    print(f"Predicted {len(terra.objects)} objects")
    
    # Display Terra
    # terra.display_terra(display_pc=True, plot_objects_on_ground=True, color_pc_clip=False)

    # Evaluate object detections
    cfg_eval = dict(cfg)
    cfg_eval.pop("terra", None)  # prevent double-passing
    obj_eval = ObjectEvaluator(args.num_imgs, terra, **cfg_eval)
    obj_eval.evaluate()