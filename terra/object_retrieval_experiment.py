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

# ---------------------------
# Transformation helpers
# ---------------------------
def load_transformation(file_path):
    transform_1d = np.load(file_path)
    trans_mat = np.eye(4)
    rot = R.from_quat(transform_1d[3:])
    trans_mat[:3, :3] = rot.as_matrix()
    trans_mat[:3, 3] = transform_1d[:3]
    return trans_mat

def project_bbox_to_image(bbox_3d, T_global_lidar, T_cam_lidar, K):
    corners_h = np.hstack([bbox_3d, np.ones((bbox_3d.shape[0],1))]).T
    X_cam = T_cam_lidar @ np.linalg.inv(T_global_lidar) @ corners_h
    x = X_cam[0,:] / X_cam[2,:]
    y = X_cam[1,:] / X_cam[2,:]
    u = K[0,0]*x + K[0,2]
    v = K[1,1]*y + K[1,2]
    return np.vstack([u.astype(int), v.astype(int)]).T

def draw_bbox_on_image(img, bbox_2d, color=(255,0,0), thickness=2):
    x_min, y_min = np.min(bbox_2d, axis=0)
    x_max, y_max = np.max(bbox_2d, axis=0)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img

# ---------------------------
# Filename parsing
# ---------------------------
def parse_camera_and_timestamp(img_path):
    img_name = os.path.basename(img_path)
    m = re.match(r"cam(\d+)_img_([0-9]+\.[0-9]+)", img_name)
    if m is None:
        raise ValueError(f"Cannot parse {img_name}")
    cam_id = int(m.group(1)) - 1  # 0-indexed
    timestamp = float(m.group(2))
    return cam_id, timestamp

def load_camera_image_timestamps(cam_folder, cam_id):
    img_paths = sorted(glob.glob(f"{cam_folder}/cam{cam_id+1}_img_*.jpg"))
    return [(float(os.path.basename(p).split("_")[-1].replace(".jpg","")), p) for p in img_paths]

def load_lidar2cam_transforms(transform_folder, cam_id):
    files = sorted(glob.glob(f"{transform_folder}/transform_lidar_to_cam{cam_id+1}_*.npy"))
    return [(float(os.path.basename(f).split("_")[-1].replace(".npy","")), f) for f in files]

def load_lidar2global_transforms(transform_folder):
    files = sorted(glob.glob(f"{transform_folder}/transform_lidar_to_map_*.npy"))
    return [(float(os.path.basename(f).split("_")[-1].replace(".npy","")), f) for f in files]

def find_closest_transform(cam_ts, transforms_list):
    times = [t[0] for t in transforms_list]
    idx = min(range(len(times)), key=lambda i: abs(times[i]-cam_ts))
    return transforms_list[idx][1]

# ---------------------------
# Visualization function
# ---------------------------
def visualize_objects(cfg, terra):
    num_cams = cfg['num_cams']
    data_folder = cfg['data_folder']

    # Camera intrinsics
    K_list = [np.array(cfg[f'cam{i}_K']).reshape(3,3) for i in range(1,num_cams+1)]

    # Image and transform folders
    cam_image_folders = [os.path.join(data_folder, f"camera{i}_images") for i in range(1,num_cams+1)]
    cam_transform_folders = [os.path.join(data_folder, f"transformations_lidar2cam{i}") for i in range(1,num_cams+1)]
    lidar2global_folder = os.path.join(data_folder, "transformations_lidar2global")

    # Preload transforms
    lidar2cam_transforms = [load_lidar2cam_transforms(f, i) for i,f in enumerate(cam_transform_folders)]
    lidar2global_transforms = load_lidar2global_transforms(lidar2global_folder)

    # KDTree for place nodes
    place_nodes = [n for n, d in terra.terra_3dsg.nodes(data=True) if d["level"] == 1]
    place_pos = np.array([terra.terra_3dsg.nodes[n]["pos"] for n in place_nodes])
    kdt = KDTree(place_pos)

    for obj_idx, tobj in enumerate(terra.objects):
        obb = tobj.get_bbox()
        obb_corners = np.asarray(obb.get_box_points())
        obb_center = obb.center[:2]

        # Nearest place node
        _, idx = kdt.query(obb_center)
        closest_place_node = place_nodes[idx]

        # Images that see this node
        img_indices = terra.nodeid_2_img_idx[closest_place_node]

        for img_idx in img_indices:
            img_path = terra.img_names[img_idx]
            cam_id, timestamp = parse_camera_and_timestamp(img_path)

            # Load image
            img_full_path = os.path.join(cam_image_folders[cam_id], os.path.basename(img_path))
            if not os.path.exists(img_full_path):
                continue
            img = cv2.imread(img_full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load closest transforms
            lidar2cam_file = find_closest_transform(timestamp, lidar2cam_transforms[cam_id])
            lidar2global_file = find_closest_transform(timestamp, lidar2global_transforms)
            T_cam_lidar = load_transformation(lidar2cam_file)
            T_global_lidar = load_transformation(lidar2global_file)

            # Project 3D bbox
            bbox_2d = project_bbox_to_image(obb_corners, T_global_lidar, T_cam_lidar, K_list[cam_id])

            # Draw bbox
            img = draw_bbox_on_image(img, bbox_2d, color=(255,0,0), thickness=2)

            # Display
            plt.figure(figsize=(8,6))
            plt.imshow(img)
            plt.title(f"Object {obj_idx} in {os.path.basename(img_path)} (place node {closest_place_node})")
            plt.axis('off')
            plt.show()

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
    terra.display_terra()

    # Visualize objects on camera images
    visualize_objects(cfg, terra)
