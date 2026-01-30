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

def draw_text_with_background(img, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=0.8, font_color=(0,0,0), bg_color=(255,255,255),
                              thickness=2, padding=5):
    """
    Draws text on an image with a filled background rectangle.
    """
    x, y = position
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Draw rectangle
    cv2.rectangle(img, (x - padding, y - text_h - padding), 
                         (x + text_w + padding, y + baseline + padding), 
                         bg_color, -1)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
    return img

def project_pt_to_image(pt, T_global_lidar, T_cam_lidar, K, h, w):
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

def draw_center_point(img, pt_2d, color=(0, 255, 0), radius=5):
    if pt_2d is None:
        return img

    h, w = img.shape[:2]
    x, y = pt_2d

    if 0 <= x < w and 0 <= y < h:
        cv2.circle(img, (x, y), radius, color, -1)  # filled circle

    return img


def project_bbox_to_image(bbox_3d, T_global_lidar, T_cam_lidar, K):
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

def draw_bbox_on_image(img, bbox_2d, color=(255,0,0), thickness=2):
    x_min, y_min = np.min(bbox_2d, axis=0)
    x_max, y_max = np.max(bbox_2d, axis=0)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img

def build_obb_edges(corners_3d, angle_eps=1e-3):
    """
    corners_3d: (8,3) array of OBB corners in ANY order
    Returns: set of frozenset({i, j}) edges
    """
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

def draw_projected_obb(img, bbox_2d, edges, valid, color=(255, 0, 0), thickness=2):
    """
    bbox_2d: (8, 2) projected 2D corners
    edges: list of (i, j) index pairs
    """
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

    print(f"Terra found {len(terra.objects)} objects!")
    for obj_idx, tobj in enumerate(terra.objects):
        obb = tobj.get_bbox()
        task_idx = tobj.get_task_idx()
        obb_corners = np.asarray(obb.get_box_points())
        edges = build_obb_edges(obb_corners)
        obb_center_xy = obb.center[:2]

        # Nearest place node
        _, idx = kdt.query(obb_center_xy)
        closest_place_node = place_nodes[idx]

        # Images that see this node
        img_indices = terra.nodeid_2_img_idx[closest_place_node]

        shown = 0
        for img_idx in img_indices:
            if shown >= 1:#20
                break

            img_path = terra.img_names[img_idx]
            cam_id, timestamp = parse_camera_and_timestamp(img_path)

            img_full_path = os.path.join(
                cam_image_folders[cam_id],
                os.path.basename(img_path)
            )
            if not os.path.exists(img_full_path):
                continue

            img = cv2.imread(img_full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load closest transforms
            lidar2cam_file = find_closest_transform(timestamp, lidar2cam_transforms[cam_id])
            lidar2global_file = find_closest_transform(timestamp, lidar2global_transforms)

            T_cam_lidar = load_transformation(lidar2cam_file)
            T_global_lidar = load_transformation(lidar2global_file)

            center_2d = project_pt_to_image(
                obb.center, T_global_lidar, T_cam_lidar, K_list[cam_id],
                img.shape[0], img.shape[1]
            )
            if center_2d is None:
                continue
            img = draw_center_point(img, center_2d)

            # Project 3D bbox
            bbox_2d, valid = project_bbox_to_image(
                obb_corners, T_global_lidar, T_cam_lidar, K_list[cam_id]
            ) # [8x2]
            img = draw_projected_obb(img, bbox_2d, edges, valid)
            
            object_prompt = cfg['object_tasks'][task_idx]["task"]
            img = draw_text_with_background(img, object_prompt, position=(10, 30))
            
            # Create figure but DO NOT show yet
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(
                f"Object {obj_idx} | {os.path.basename(img_path)} | place {closest_place_node}"
            )
            plt.axis("off")

            shown += 1

        # 🔑 ONE blocking call per object
        if shown > 0:
            plt.show()

# ---------------------------
# Main
# --------------------q-------
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
