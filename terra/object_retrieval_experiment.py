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

import clip

from terra_utils import load_terra

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
        self.K_list = [np.array(kwargs[f'cam{i}_K']).reshape(3,3) for i in range(1,self.num_cams+1)]
        
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
        self.gt_matches = [0 for _ in range(self.num_tasks)] # M_t
        self.task_rel_count = [0 for _ in range(self.num_tasks)] # \bar{N}_t
        self.task_rel_matches = [0 for _ in range(self.num_tasks)] # \bar{M}_t
    
    def evaluate(self):
        self.record_recall_metrics()
        self.record_precision_metrics()
        
        # Print out metrics
        recall = sum(self.gt_matches) / sum(self.gt_obj_count)
        precision = sum(self.task_rel_matches) / sum(self.task_rel_count)
        F1 = 0.0 if (recall + precision) == 0 else 2 * (precision * recall) / (precision + recall)
        
        print(f"\n\nOBJECT METRICS FOR DATASET: {self.data_folder}")
        print(f"Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {F1:.3f}\n\n")
    
    def record_precision_metrics(self):
        print("\n\nCalculating Precision Metrics:")
        Nbar_t = self.get_top90percent_objs()
        for t in range(self.num_tasks):
            self.task_rel_count[t] = len(Nbar_t[t])
            Mbar_t = self.count_matches(Nbar_t[t])
            self.task_rel_matches[t] = Mbar_t            
            print(f"Task {self.task_names[t]}: GT Count = {len(Nbar_t[t])}, GT Matches = {Mbar_t}")
            
    def record_recall_metrics(self):
        print("\n\nCalculating Recall Metrics:")
        for t in range(self.num_tasks):
            Nt = self.gt_obj_count[t]
            topk_objs = self.get_topk_objs_for_task(t, Nt)
            Mt = self.count_matches(topk_objs, task_idx=t)
            self.gt_matches[t] = Mt
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
                if score >= (0.9 * max_task_scores[t]):
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

            # Images that see this node
            img_indices = self.terra.nodeid_2_img_idx[closest_place_node]

            shown = 0
            for img_idx in img_indices:
                if shown >= self.display_num_imgs:
                    break

                img_path = self.terra.img_names[img_idx]
                cam_id, timestamp = self.parse_camera_and_timestamp(img_path)

                img_full_path = os.path.join(
                    self.cam_image_folders[cam_id],
                    os.path.basename(img_path)
                )
                if not os.path.exists(img_full_path):
                    continue

                img = cv2.imread(img_full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Load closest transforms
                lidar2cam_file = self.find_closest_transform(timestamp, self.lidar2cam_transforms[cam_id])
                lidar2global_file = self.find_closest_transform(timestamp, self.lidar2global_transforms)

                T_cam_lidar = self.load_transformation(lidar2cam_file)
                T_global_lidar = self.load_transformation(lidar2global_file)

                # Project 3D bbox
                bbox_2d, valid = self.project_box_3dto2d(
                    obb_corners, T_global_lidar, T_cam_lidar, self.K_list[cam_id]
                ) # [8x2]
                
                center_2d = self.project_point_3dto2d(
                    obb.center, T_global_lidar, T_cam_lidar, self.K_list[cam_id],
                    img.shape[0], img.shape[1]
                )
                if center_2d is None:
                    continue
                
                ## Draw on image
                img = self.draw_center_point(img, center_2d)
                img = self.draw_projected_obb(img, bbox_2d, edges, valid)
                img = self.draw_task(img, self.task_names[task_idx], position=(10, 30))
                
                # Create figure but DO NOT show yet
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.title(
                    f"Object {obj_idx} | {os.path.basename(img_path)} | place {closest_place_node}"
                )
                plt.axis("off")

                shown += 1

            if shown > 0:
                decision = self._wait_for_yn()

                if decision is None:
                    assert False, f"[Object {obj_idx}] for task {self.task_names[task_idx]}. No decision made!"
                elif decision:
                    print(f"[Object {obj_idx}] for task {self.task_names[task_idx]}. Matched (y)")
                    num_matches += 1
                else:
                    print(f"[Object {obj_idx}] for task {self.task_names[task_idx]}. Not a match (n)")
            # else:
            #     assert False, "No images see bounding box!"
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
                plt.close(event.canvas.figure)

        fig = plt.gcf()
        cid = fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()  # blocks
        fig.canvas.mpl_disconnect(cid)

        return decision["value"]
    
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

    def project_point_3dto2d(self, pt, T_global_lidar, T_cam_lidar, K, h, w):
        pt_h = np.array([pt[0], pt[1], pt[2], 1.0])  # (4,)
        cam_pt = (T_cam_lidar @ np.linalg.inv(T_global_lidar) @ pt_h)[:3]

        z = cam_pt[2]
        if z <= 1e-6:
            return None  # behind camera

        proj = K @ cam_pt
        x = int(proj[0] / z)
        y = int(proj[1] / z)

        if x < 0 or x >= w or y < 0 or y >= h:
            return None 
        
        return np.array([x, y])

    def project_box_3dto2d(self, bbox_3d, T_global_lidar, T_cam_lidar, K):
        corners_h = np.hstack([bbox_3d, np.ones((bbox_3d.shape[0],1))]).T
        cam_points_h_wide = T_cam_lidar @ np.linalg.inv(T_global_lidar) @ corners_h # [4x8]
        cam_points_h = cam_points_h_wide.T # [8x4] 
        cam_points = cam_points_h[:,:3] # [8x3]
        
        # Keep points in front of camera
        zs = cam_points[:, 2]
        valid = zs > 1e-6
        # cam_points = cam_points[valid]
        # zs = zs[valid]
        
        proj_pts = cam_points @ K.T
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
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load Terra
    terra = load_terra(cfg['terra'])
    terra.alpha = cfg['alpha']

    # Setup CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/16", device=device)

    # Encode tasks
    tasks = [task["task"] for task in cfg['object_tasks']]
    tasks[:0] = terra.terrain_names
    input_task_embs = [clip_model.encode_text(clip.tokenize([t]).to(device)).float() for t in tasks]
    input_task_tensor = torch.vstack(input_task_embs)

    # Predict objects
    terra.predict_objects(input_task_tensor, tasks[terra.num_terrain:], cfg['prediction_method'])
    
    # Display Terra
    # terra.display_terra()

    # Evaluate object detections
    cfg_eval = dict(cfg)
    cfg_eval.pop("terra", None)  # prevent double-passing
    obj_eval = ObjectEvaluator(1, terra, **cfg_eval)
    obj_eval.evaluate()