import os
import re
import yaml
from argparse import ArgumentParser
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from PIL import Image
import cv2
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import pickle as pkl
import torch
import torch.nn.functional as F
from ultralytics import YOLO, YOLOE, FastSAM
import clip
from utils import tensor_cosine_similarity, numeric_key

class MS_Map:
    def __init__(self, args):
        ## Processing parameters
        self.data_folder = args['data_folder']
        self.num_cams = args['num_cams']
        self.scan_step_sz = args['save_step_size']
        self.continue_processing = args['continue_processing']
        self.DEBUG_MODE = args['debug']
        if self.DEBUG_MODE:
            print("Running in DEBUG_MODE")
        self.unaligned_threshold = args['unaligned_threshold']
        self.cam_axis = args['cam_axis'] # ['+x', '-x', '+y', '-y']
        
        #######################
        ## Define Parameters ##
        #######################
        self.K = np.array(args['cam_K']).reshape(3,3)
        self.theta_cos_sim = args['match_threshold'] # cos_sim threshold to determine a match
        self.use_dbscan = args['use_dbscan']
        self.dbscan_global = DBSCAN(eps=args['dbscan_eps'], min_samples=args['dbscan_min_samples']) # change depending on voxel sizes of global PC
        self.cam2point_dist_thresh = args['cam2point_dist_threshold'] # Include images within 20 meters of points
        self.num_scans = 0

        # Timing Lists TODO: Delete when done
        self.scan_times = []
        self.pcl_in_image_times = []
        self.yolo_times = []
        self.fastsam_clip_times = []
        self.segment_with_fastsam_times = []
        self.extract_mask_embs_times = []

        ##########################
        ## Load Models and Data ##
        ##########################

        # Load Models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        model_path = '/FastSAM.pt'
        self.fastsam_model = FastSAM(model_path)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device)
        self.logit_scale = self.clip_model.logit_scale.exp()
        
        self.use_yoloe = args['use_yoloe']
        if self.use_yoloe:
            self.yolo_model = YOLOE("yoloe-11s-seg.pt")
            self.yolo_model.set_classes(args['terrain_prompts'], self.yolo_model.get_text_pe(args['terrain_prompts']))
        else:
            self.yolo_model = YOLO(args['yolo_model_path']) # orig_and_sunnyfeb_11nano_640sz/weights/best.pt
        self.yolo_conf_thresh = args['yolo_conf_threshold']
        
        # Load global point cloud
        global_pc_folder = os.path.join(self.data_folder, "global_pc")
        global_pc_files = sorted(Path(global_pc_folder).glob("*.npy"),  key=numeric_key)        
        latest_global_pc_file = global_pc_files[-1] # global_map_idx 33 for sunny midday data
        self.global_pc = np.load(latest_global_pc_file)
        self.global_kdtree = KDTree(self.global_pc[:, :3])  # Dictionary to hold matching global points using the x, y, z columns

        # Load time-synced sensor data and transformation files
        self.lidar_pc_folder = os.path.join(self.data_folder, "lidar_pc")
        self.camera_folders = []
        self.transforms_lidar2cam_folders = []
        for nc in range(self.num_cams):
            self.camera_folders.append(os.path.join(self.data_folder, f"camera{nc+1}_images"))
            self.transforms_lidar2cam_folders.append(
                os.path.join(self.data_folder, f"transformations_lidar2cam{nc+1}")
            )
        self.transforms_lidar2global_folder = os.path.join(self.data_folder, "transformations_lidar2global")

        #Load timesteps
        self.camera_timestamps_list = []
        for folder in self.camera_folders:
            timestamps = []
            for fn in Path(folder).glob("*.jpg"):
                ts = float(fn.stem.split("_")[-1])
                timestamps.append(ts)
            timestamps.sort()
            self.camera_timestamps_list.append(timestamps)
        
        self.lidar_pc_timestamps = []
        for lidar_pc_file in Path(self.lidar_pc_folder).glob("*.npy"):
            # Assuming filename format: lidar_pc_{timestamp}.jpg
            timestamp = float(lidar_pc_file.stem.split("_")[-1])  # Extract timestamp as float
            self.lidar_pc_timestamps.append(timestamp)
        self.lidar_pc_timestamps.sort()  # Sort timestamps for easier matching
        
        self.tf_l2c_timestamps_list = []
        for folder in self.transforms_lidar2cam_folders:
            timestamps = []
            for fn in Path(folder).glob("*.npy"):
                ts = float(fn.stem.split("_")[-1])
                timestamps.append(ts)
            timestamps.sort()
            self.tf_l2c_timestamps_list.append(timestamps)

    def make_map(self):
        t_begin = time.time()

        if self.continue_processing:
            print("Picking up where you last left off")
            self.load_last_saved_data()
        else:
            print("Starting at the beginning")
            #Embed yolo class names with CLIP
            yolo_clip_embs = [] 
            yolo_classes = self.yolo_model.names
            for cls_idx, cls_str in yolo_classes.items():
                clip_emb = self.clip_model.encode_text(clip.tokenize([cls_str]).to(self.device)).float()
                yolo_clip_embs.append(clip_emb)

            #Initialize data structures
            self.clip_tensor = torch.vstack(yolo_clip_embs) # (num_classes, 512)
            self.pc_dict = {} #defaultdict(lambda: defaultdict(int))  # {point_index: {clip_id: count}}
            self.num_scans = 0
            self.img_clips = []
            self.saved_img_names = []
            self.map_globalidx2imgidx = {} # map from global_index to img_index {g_idx: set(0,34,2), ...}
            self.map_globalidx2imgidx_nodistthresh = {} # map from global_index to img_index {g_idx: set(0,34,2), ...}
            self.map_globalidx2dist_nodistthresh = {} # map from global_index to img_index {g_idx: dist_m, ...}


        #Iterate through each scan
        for scan_idx, transform_lidar2global_file in enumerate(sorted(Path(self.transforms_lidar2global_folder).glob("*.npy"), key=numeric_key)):
            if self.continue_processing and scan_idx <= self.last_scan_idx:
                continue # Skip over already processed scans

            itr_t0 = time.time()
            timestamp = transform_lidar2global_file.stem.split("_")[-1]  # Assuming format lidar_pc_{timestamp}.npy
            print(f"\nScan Index {scan_idx} at timestamp {timestamp}\n")
            
            ## Ensure files are available/time synced for this timestamp ##
            closest_lidar_pc_timestamp = min(self.lidar_pc_timestamps, key=lambda x: abs(x - float(timestamp)))
            if abs(closest_lidar_pc_timestamp - float(timestamp)) > self.unaligned_threshold:
                print("lidar scan unaligned")
                continue
            lidar_pc_file = os.path.join(self.lidar_pc_folder, f'lidar_pc_{closest_lidar_pc_timestamp}.npy') #TODO Change this back to 4 decimal places
            
            camera_image_files = []
            for nc, (cam_timestamps, folder) in enumerate(zip(self.camera_timestamps_list, self.camera_folders)):
                closest_ts = min(cam_timestamps, key=lambda x: abs(x - float(timestamp)))
                if abs(closest_ts - float(timestamp)) > self.unaligned_threshold:
                    print("Camera stream unaligned")
                    continue
                camera_image_files.append(os.path.join(folder, f"cam{nc+1}_img_{closest_ts}.jpg"))
                
            transform_lidar_to_cam_files = []
            for nc, (tf_timestamps, folder) in enumerate(zip(self.tf_l2c_timestamps_list, self.transforms_lidar2cam_folders)):
                closest_ts = min(tf_timestamps, key=lambda x: abs(x - float(timestamp)))
                if abs(closest_ts - float(timestamp)) > self.unaligned_threshold:
                    print("TF lidar-to-camera unaligned")
                    continue
                transform_lidar_to_cam_files.append(os.path.join(folder, f"transform_lidar_to_cam{nc+1}_{closest_ts}.npy"))
            
            if len(camera_image_files) == 0 or len(transform_lidar_to_cam_files) == 0:
                continue
            
            ## Load lidar Data ##
            self.load_lidar_data(lidar_pc_file, camera_image_files, transform_lidar_to_cam_files, transform_lidar2global_file)

            ## CLIP vector of base image ##
            self.clip_base_image(camera_image_files)

            ## Keep only 3D Points in Image ##
            pcl_in_image_t0 = time.time()
            self.get_pcl_points_found_in_image(scan_idx)
            pcl_in_image_t1 = time.time()
            self.pcl_in_image_times.append(pcl_in_image_t1 - pcl_in_image_t0)

            ## Yolo Segmentation ##
            yolo_t0 = time.time()
            self.yolo_segmentation(scan_idx)
            yolo_t1 = time.time()
            self.yolo_times.append(yolo_t1 - yolo_t0)

            ## FastSAM+CLIP ##
            self.fastsam_and_clip(scan_idx)

            ## Save intermediate results every scan_step_sz ##
            if (scan_idx % self.scan_step_sz == 0):
                self.save_semantic_pcl(scan_idx)

                # Show intermediate results if in debug mode
                if (self.DEBUG_MODE):
                    self.display_global_pcl()

            itr_t1 = time.time()
            self.num_scans = scan_idx
            print(f"Iteration {scan_idx} runtime:",itr_t1 - itr_t0,"sec")
            self.scan_times.append(itr_t1 - itr_t0)

        print("Average scan time:", np.mean(self.scan_times))
        print("Average point cloud extraction time:", np.mean(self.pcl_in_image_times))
        print("Average YOLO time:", np.mean(self.yolo_times))
        print("Average FastSam Segmentation time:", np.mean(self.segment_with_fastsam_times))
        print("Average Extract Mask Embeddings time:", np.mean(self.extract_mask_embs_times))

        t_end = time.time()
        print("Finished iterating through all lidar scans in",t_end-t_begin,"seconds for",self.num_scans + 1,"scans")

        ## Save final results ##
        self.save_semantic_pcl(self.num_scans)

        ## Visualize final global map colored by class ##
        self.display_global_pcl()

    def load_lidar_data(self, lidar_pc_file, camera_image_files, transform_lidar_to_cam_files, transform_lidar_to_global_file):
        # Load lidar point cloud data
        self.lidar_pc = np.load(lidar_pc_file)
    
        # Load all images
        self.camera_images = [cv2.imread(f) for f in camera_image_files]
        self.IMG_H, self.IMG_W = self.camera_images[0].shape[:2]

        # Load transformation data
        self.transforms_lidar_to_cam = [self.load_transformation(tf_l2c_file) for tf_l2c_file in transform_lidar_to_cam_files]
        self.transform_lidar_to_global = self.load_transformation(transform_lidar_to_global_file)

    def clip_base_image(self, camera_image_files):
        self.img_clips = []
        for cam_idx, img in enumerate(self.camera_images):
            prep = self.clip_preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.img_clips.append(self.clip_model.encode_image(prep))
            self.saved_img_names.append(camera_image_files[cam_idx])

    def get_pcl_points_found_in_image(self, scan_idx):
        # Get the 3D point cloud points that correspond to the 2D image
        # This can be done using the camera intrinsics and the depth information
        self.map_yx2idx = [dict() for _ in range(self.num_cams)] # map from 2D image coord to 3D PC index
        self.map_lidar2globalidx = {} # map from 3D lidar PC index to 3D global PC index
        self._pixel_to_global_idx = [
            np.full((self.IMG_H, self.IMG_W), -1, dtype=np.int32)
            for _ in range(self.num_cams)
        ]
        lidar_imgs = [
            np.zeros((self.IMG_H,self.IMG_W))
            for _ in range(self.num_cams)
        ]
        
        self.lidar_imgs_bool = []
        for cam_idx, img in enumerate(self.camera_images):
            # Filter: Keep only points in front of the camera/lidar
            if self.cam_axis[cam_idx] == '+x':
                mask = self.lidar_pc[:, 0] >= 0
            elif self.cam_axis[cam_idx] == '-x':
                mask = self.lidar_pc[:, 0] < 0
            elif self.cam_axis[cam_idx] == '+y':
                mask = self.lidar_pc[:, 1] >= 0
            elif self.cam_axis[cam_idx] == '-y':
                mask = self.lidar_pc[:, 1] < 0
            else:
                print("Incorrect cam_axis argument. Must be either [+x, -x, +y, -y]")
                exit()
            points = self.lidar_pc[mask]
            
            # Split into geometry + intensity
            points_xyz = points[:, :3]                # (N,3)
            intensity = points[:, 3]                  # (N,)

            #Add homogeneous coordinate (N, 4)
            points_h = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])

            # Transform LiDAR -> camera frame (N, 4)
            cam_points_h = points_h @ self.transforms_lidar_to_cam[cam_idx].T

            # Drop homogeneous coord (N, 3)
            cam_points = cam_points_h[:, :3]

            # Project onto image plane (N, 3)
            proj_points = cam_points @ self.K.T # assumes same intrinsic matrix for all cams

            # Normalize by z (N,) and round to nearest pixel
            zs = proj_points[:, 2]
            xs = np.round(proj_points[:, 0] / zs).astype(int)
            ys = np.round(proj_points[:, 1] / zs).astype(int)

            # Original indices (needed for map lookups)
            pt_indices = np.nonzero(mask)[0]

            # Filter which points are in the image bounds
            in_bounds = (xs >= 0) & (xs < self.IMG_W) & (ys >= 0) & (ys < self.IMG_H)
            xs, ys = xs[in_bounds], ys[in_bounds]
            pt_indices = pt_indices[in_bounds]
            points_xyz = points_xyz[in_bounds]
            intensity = intensity[in_bounds]

            # LiDAR → global (N,3), avoids np.append per point
            p_Gs = points_xyz @ self.transform_lidar_to_global[:3, :3].T + self.transform_lidar_to_global[:3, 3]

            # Batch KD-tree query
            dists, g_indices = self.global_kdtree.query(p_Gs)

            for x, y, pt_idx, inten, g_idx, p_L_dist in zip(xs, ys, pt_indices, intensity, g_indices, np.linalg.norm(points_xyz, axis=1)):
                self.map_yx2idx[cam_idx][(y, x)] = pt_idx
                lidar_imgs[cam_idx][y, x] = inten  # use intensity

                # Nearest global point
                self.map_lidar2globalidx[pt_idx] = g_idx
                self._pixel_to_global_idx[cam_idx][y, x] = g_idx

                # --- Within distance threshold ---
                if p_L_dist < self.cam2point_dist_thresh:
                    if g_idx in self.map_globalidx2imgidx:
                        self.map_globalidx2imgidx[g_idx].add(len(self.saved_img_names) - 1 - (2 - cam_idx))
                    else:
                        self.map_globalidx2imgidx[g_idx] = {len(self.saved_img_names) - 1 - (2 - cam_idx)}

                # --- No distance threshold ---
                if g_idx not in self.map_globalidx2imgidx:
                    if g_idx in self.map_globalidx2dist_nodistthresh:
                        if p_L_dist < self.map_globalidx2dist_nodistthresh[g_idx]:
                            self.map_globalidx2dist_nodistthresh[g_idx] = p_L_dist
                            self.map_globalidx2imgidx_nodistthresh[g_idx].add(len(self.saved_img_names) - 1 - (2 - cam_idx))
                    else:
                        self.map_globalidx2dist_nodistthresh[g_idx] = p_L_dist
                        self.map_globalidx2imgidx_nodistthresh[g_idx] = {len(self.saved_img_names) - 1 - (2 - cam_idx)}

            lidar_imgs[cam_idx] = lidar_imgs[cam_idx] / np.max(lidar_imgs[cam_idx])
            self.lidar_imgs_bool.append(lidar_imgs[cam_idx].astype(bool)) # True for non-zero intensities
            if self.DEBUG_MODE and (scan_idx % self.scan_step_sz == 0):
                self.display_image("Lidar Image", lidar_imgs[cam_idx])

    def yolo_segmentation(self, scan_idx):
        self.yolo_results = []
        self.yolo_masks = []
        for cam_idx, img in enumerate(self.camera_images):
            result = self.yolo_model(img, conf=self.yolo_conf_thresh)
            self.yolo_results.append(result)
            if self.DEBUG_MODE and (scan_idx % self.scan_step_sz == 0):
                img_with_masks = result[0].plot(
                    labels=True,
                    boxes=True,
                    masks=True,
                    probs=False,
                    conf=False,
                    show=False,
                    save=False,
                )
                self.display_image("YOLO Segmentation", img_with_masks)
                
            if self.yolo_results[cam_idx][0].masks is not None:
                self.yolo_masks.append(self.get_yolo_class_masks(img, self.yolo_results[cam_idx]))

                ## Associate Global Points to CLIP embs ##
                class_name = self.yolo_model.names
                for cls_id, mask in self.yolo_masks[cam_idx].items():
                    filtered_lidar_img = np.bitwise_and(self.lidar_imgs_bool[cam_idx], mask.astype(bool))

                    if self.DEBUG_MODE and (scan_idx % self.scan_step_sz == 0):
                        self.display_image(f"Filtered LiDAR Image {class_name[cls_id]}", (filtered_lidar_img.astype(np.uint8)*255))

                    y_indices, x_indices = np.where(filtered_lidar_img > 0)
                    for y, x in zip(y_indices, x_indices):
                        g_idx = self.map_lidar2globalidx[self.map_yx2idx[cam_idx][(y,x)]]
                        if g_idx in self.pc_dict:
                            if cls_id in self.pc_dict[g_idx]:
                                self.pc_dict[g_idx][cls_id] += 1
                            else:
                                self.pc_dict[g_idx][cls_id] = 1
                        else:
                            self.pc_dict[g_idx] = {cls_id: 1}
            else:
                self.yolo_masks.append([])

    def fastsam_and_clip(self, scan_idx):
        """Main FastSAM + CLIP pipeline."""
        for cam_idx, img in enumerate(self.camera_images):
            filtered_image = self.prepare_filtered_image(cam_idx, scan_idx)

            segment_with_fastsam_t0 = time.time()
            fastsam_masks = self.segment_with_fastsam(filtered_image, scan_idx)
            segment_with_fastsam_t1 = time.time()
            self.segment_with_fastsam_times.append(segment_with_fastsam_t1 - segment_with_fastsam_t0)

            extract_mask_embs_t0 = time.time()
            clip_embs_tensor, global_idxs = self.extract_mask_embeddings_and_indices(cam_idx, fastsam_masks, filtered_image)
            extract_mask_embs_t1 = time.time()
            self.extract_mask_embs_times.append(extract_mask_embs_t1 - extract_mask_embs_t0)

            if clip_embs_tensor.shape[0] > 0:  # Only update if we have valid embeddings
                self.update_clip_and_pc_dict(clip_embs_tensor, global_idxs)

    def prepare_filtered_image(self, cam_idx, scan_idx):
        """
        Removes YOLO masks from the lidar image (if present) to prepare for FastSAM segmentation.
        Optionally displays the debug image.
        Returns:
            filtered_image (np.ndarray): Image after mask removal.
        """
        if self.yolo_results[cam_idx][0].masks is not None:
            filtered_image = self.remove_yolo_masks(self.camera_images[cam_idx], self.yolo_masks[cam_idx])
        else:
            filtered_image = self.camera_images[cam_idx]

        if self.DEBUG_MODE and (scan_idx % self.scan_step_sz == 0):
            self.display_image("Removed YOLO Masks", filtered_image)

        return filtered_image

    def segment_with_fastsam(self, filtered_image, scan_idx):
        """
        Segments the image using FastSAM and returns the boolean masks.
        Returns:
            fastsam_masks (torch.Tensor): Boolean tensor of segmentation masks.
        """
        fastsam_results = self.fastsam_model(
            filtered_image,
            device=self.device,
            retina_masks=True,
            imgsz=filtered_image.shape[:2],
            conf=0.003,
            iou=0.25,
            max_det=100,
        )

        if self.DEBUG_MODE and (scan_idx % self.scan_step_sz == 0):
            mask_img = fastsam_results[0].plot(
                conf=False,
                labels=False,
                boxes=False,
                probs=False,
                color_mode="instance",
            )
            self.display_image("FastSAM Masks", mask_img)

        fastsam_masks = fastsam_results[0].masks.data.bool()

        if fastsam_masks.numel() == 0:
            print("NO MASKS FOUND")
            exit()

        return fastsam_masks

    def extract_mask_embeddings_and_indices(self, cam_idx, fastsam_masks, filtered_image):
        """
        Processes each FastSAM mask:
            - Filters out terrain masks
            - Crops and stores for later CLIP batching
            - Filters by lidar points and optionally applies DBSCAN
        Returns:
            fastsam_clip_embs_tensor (torch.Tensor): Embeddings for each valid mask.
            fastsam_global_idxs (list[list[int]]): List of point cloud indices for each mask.
        """

        buffer = 10
        kernel = np.ones((5, 5), np.uint8)

        # --- Precompute boolean masks for faster overlap checks ---
        camera_nonzero = np.any(self.camera_images[cam_idx] != 0, axis=2)    # HxW bool
        filtered_nonzero = np.any(filtered_image != 0, axis=2)   # HxW bool

        clip_input_list = []
        valid_masks_global_idxs = []

        for mask in fastsam_masks:
            mask_bool = (mask.cpu().numpy() > 0)
            if mask_bool.sum() == 0:
                continue

            # overlap checks
            cam_overlap = np.count_nonzero(mask_bool & camera_nonzero)
            if cam_overlap == 0:
                continue
            filtered_overlap = np.count_nonzero(mask_bool & filtered_nonzero)
            if filtered_overlap / float(mask_bool.sum()) < 0.5:
                continue

            # bounding box and dilation
            mask_u8 = (mask_bool.astype(np.uint8) * 255)
            x, y, w, h = cv2.boundingRect(mask_u8)
            x1, y1 = max(0, x - buffer), max(0, y - buffer)
            x2, y2 = min(self.IMG_W, x + w + buffer), min(self.IMG_H, y + h + buffer)


            dilated = cv2.dilate(mask_u8, kernel, iterations=3).astype(bool)
            if not dilated.any():
                continue

            # crop and zero-out background inside crop
            cropped_bb = filtered_image[y1:y2, x1:x2].copy()
            roi = dilated[y1:y2, x1:x2]
            cropped_bb[~roi] = 0
            try:
                pil_crop = Image.fromarray(cropped_bb)
            except Exception:
                pil_crop = Image.fromarray((np.clip(cropped_bb, 0, 255)).astype(np.uint8))

            # vectorized lookup for mapped lidar pixels inside the dilated mask
            # ys, xs = np.nonzero(dilated & self.lidar_img_bool)
            ys, xs = np.nonzero(dilated & filtered_nonzero & self.lidar_imgs_bool[cam_idx])
            if ys.size == 0:
                continue
            candidate_global_idxs = self._pixel_to_global_idx[cam_idx][ys, xs]
            # filter out unmapped entries (-1)
            candidate_global_idxs = candidate_global_idxs[candidate_global_idxs >= 0]
            if candidate_global_idxs.size == 0:
                continue

            # --- DBSCAN per-mask (reverted to original behavior) ---
            if self.use_dbscan:
                # gather 3D points corresponding to candidate_global_idxs
                corresponding_global_pc_arr = self.global_pc[candidate_global_idxs, :3]
                if corresponding_global_pc_arr.size == 0:
                    continue

                # fit_predict on this mask's points
                labels = self.dbscan_global.fit_predict(corresponding_global_pc_arr)
                # find non-noise clusters and pick the largest cluster
                unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                if counts.size == 0 or counts.max() == 0:
                    continue
                largest_label = unique_labels[np.argmax(counts)]
                # select global indices belonging to that largest cluster
                largest_global_pt_indices = candidate_global_idxs[labels == largest_label].tolist()
            else:
                largest_global_pt_indices = candidate_global_idxs.tolist()

            if not largest_global_pt_indices:
                continue

            clip_input_list.append(pil_crop)
            valid_masks_global_idxs.append(largest_global_pt_indices)

        if len(clip_input_list) == 0:
            return torch.empty((0, self.clip_tensor.shape[1])), []

        # preprocess and batch for CLIP
        preprocessed = [self.clip_preprocess(img).unsqueeze(0) for img in clip_input_list]
        clip_input_batch = torch.cat(preprocessed, dim=0).to(self.device, non_blocking=True)

        with torch.no_grad():
            fastsam_clip_embs_tensor = self.clip_model.encode_image(clip_input_batch)

        return fastsam_clip_embs_tensor, valid_masks_global_idxs

    def update_clip_and_pc_dict(self, clip_embs_tensor, global_idxs):
        """
        Compares embeddings to existing CLIP tensor and updates:
            - CLIP embeddings
            - Point cloud dictionary (pc_dict)
        """
        scores = tensor_cosine_similarity(clip_embs_tensor, self.clip_tensor)

        for mask_idx in range(scores.shape[0]):
            if scores[mask_idx, :].max().item() > self.theta_cos_sim:
                max_clip_id = scores[mask_idx, :].argmax().item()
            else:
                self.clip_tensor = torch.cat([self.clip_tensor, clip_embs_tensor[mask_idx, :].unsqueeze(0)], dim=0)
                max_clip_id = self.clip_tensor.shape[0] - 1

            for g_idx in global_idxs[mask_idx]:
                self.pc_dict.setdefault(g_idx, {}).setdefault(max_clip_id, 0)
                self.pc_dict[g_idx][max_clip_id] += 1

    def save_semantic_pcl(self, itr):
        ## Save Semantic Point Cloud
        with open(self.data_folder+f"/ptxpt_pc_dict_itr{itr}.pkl", "wb") as f:
            pkl.dump(self.pc_dict, f)
        torch.save(self.clip_tensor,self.data_folder+f"/ptxpt_clip_tensor_itr{itr}.pt")
        with open(self.data_folder+f"/ptxpt_gidx2imgidx_{self.cam2point_dist_thresh}m_dist_dict_itr{itr}.pkl", "wb") as f:
            pkl.dump(self.map_globalidx2imgidx, f)
        with open(self.data_folder+f"/ptxpt_gidx2imgidx_no_dist_dict_itr{itr}.pkl", "wb") as f:
            pkl.dump(self.map_globalidx2imgidx_nodistthresh, f)
        with open(self.data_folder+f"/ptxpt_gidx2dist_no_dist_dict_itr{itr}.pkl", "wb") as f:
            pkl.dump(self.map_globalidx2dist_nodistthresh, f)
        img_clip_tensor = torch.vstack(self.img_clips)
        torch.save(img_clip_tensor,self.data_folder+f"/img_clip_tensor_itr{itr}.pt")
        with open(self.data_folder+f"/saved_img_names_itr{itr}.pkl","wb") as f:
            pkl.dump(self.saved_img_names, f)

        print(f"\nSaved pc_dict and clip_tensor at iteration {itr}\n")

    def display_image(self, window_name, image):
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_global_pcl(self):
        # Visualize the global point cloud colored by class
        distinct_colors = [[1,0,0],[0,1,0],[0,0,1],[1,0.5,0],[1,0,1], [0,1,1]]
        chosen_colors = distinct_colors[:len(self.yolo_model.names)]
        count_threshold = 1
        global_pts = {}

        # Map global point indices to their classes
        for global_idx in range(self.global_pc.shape[0]):
            if global_idx in self.pc_dict.keys():
                max_class, max_count = max(self.pc_dict[global_idx].items(), key=lambda x: x[1])
                # Make sure max_count is more than some threshold
                if max_count < count_threshold:
                    if -1 in global_pts.keys():
                        global_pts[-1].append(global_idx)
                    else:
                        global_pts[-1] = [global_idx]    
                elif max_class in global_pts.keys():
                    global_pts[max_class].append(global_idx)
                else:
                    global_pts[max_class] = [global_idx]
            else:
                if -1 in global_pts.keys():
                    global_pts[-1].append(global_idx)
                else:
                    global_pts[-1] = [global_idx]

        # Create point clouds for each class
        pcds = []
        for class_id in global_pts.keys():
            pcd = o3d.geometry.PointCloud()
            if class_id == -1:
                pcd.points = o3d.utility.Vector3dVector(self.global_pc[global_pts[class_id],:3])
                pcd.paint_uniform_color([0.5,0.5,0.5])
            else:
                if class_id < len(chosen_colors):
                    pcd.points = o3d.utility.Vector3dVector(self.global_pc[global_pts[class_id], :3])
                    pcd.paint_uniform_color(chosen_colors[class_id])
                else:
                    # Otherwise, generate a random color
                    random_col = self.random_color()
                    pcd.points = o3d.utility.Vector3dVector(self.global_pc[global_pts[class_id], :3])
                    pcd.paint_uniform_color(random_col)
            pcds.append(pcd)
        o3d.visualization.draw_geometries(pcds)

    def load_last_saved_data(self):
        # Find the index of the last saved data
        start_string = "ptxpt_pc_dict_itr"
        matching_files = [f for f in os.listdir(self.data_folder) if f.startswith(start_string)]
        numbers = []
        for f in matching_files:
            match = re.search(rf"{re.escape(start_string)}(\d+)", f)
            if match:
                numbers.append(int(match.group(1)))
        self.last_scan_idx = max(numbers) if numbers else None
        print("Last scan:",self.last_scan_idx)
        
        ## Load saved data
        with open(self.data_folder+f"/ptxpt_pc_dict_itr{self.last_scan_idx}.pkl", "rb") as f:
            self.pc_dict = pkl.load(f)
        self.clip_tensor = torch.load(self.data_folder+f"/ptxpt_clip_tensor_itr{self.last_scan_idx}.pt")
        self.clip_tensor = self.clip_tensor.to(self.device)
        with open(self.data_folder+f"/ptxpt_gidx2imgidx_{self.cam2point_dist_thresh}m_dist_dict_itr{self.last_scan_idx}.pkl", "rb") as f:
            self.map_globalidx2imgidx = pkl.load(f)
        with open(self.data_folder+f"/ptxpt_gidx2imgidx_no_dist_dict_itr{self.last_scan_idx}.pkl", "rb") as f:
            self.map_globalidx2imgidx_nodistthresh = pkl.load(f)
        with open(self.data_folder+f"/ptxpt_gidx2dist_no_dist_dict_itr{self.last_scan_idx}.pkl", "rb") as f:
            self.map_globalidx2dist_nodistthresh = pkl.load(f)
        self.img_clip_tensor = torch.load(self.data_folder+f"/img_clip_tensor_itr{self.last_scan_idx}.pt")
        self.img_clip_tensor = self.img_clip_tensor.to(self.device)
        self.img_clips = list(self.img_clip_tensor.unbind(dim=0))
        with open(self.data_folder+f"/saved_img_names_itr{self.last_scan_idx}.pkl","rb") as f:
            self.saved_img_names = pkl.load(f)
        self.num_scans = self.last_scan_idx + 1

    ######################
    ## Helper Functions ##
    ######################
    @staticmethod
    def random_color():
        return np.random.rand(3).tolist()  # Generates a random RGB color

    @staticmethod
    def remove_yolo_masks(img, yolo_masks):     
        IMG_H, IMG_W = img.shape[0], img.shape[1]
        
        # Create a single binary mask covering all terrain segments
        combined_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        for cls_id, mask in yolo_masks.items():
            combined_mask = np.maximum(combined_mask, mask)

        # Expand the mask to cover a larger area
        kernel = np.ones((5, 5), np.uint8)  # Larger kernel for stronger dilation
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        # Smooth edges with Gaussian blur
        blurred_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)

        # Apply inverted blurred mask to remove terrain
        non_terrain_image = cv2.bitwise_and(img, img, mask=255 - blurred_mask)
        
        return non_terrain_image

    @staticmethod
    def get_yolo_class_masks(img, yolo_results):
        IMG_H, IMG_W = img.shape[0], img.shape[1]
        
        # Dictionary to store binary masks for each YOLO class
        class_masks = {}

        # Iterate over all YOLO masks and classes
        for mask, cls in zip(yolo_results[0].masks.data, yolo_results[0].boxes.cls):
            class_id = int(cls.item())  # Get class id
            
            # Convert mask to binary and resize it to match the image dimensions
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # Convert mask to 0-255
            mask_resized = cv2.resize(mask_np, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            # Create a binary mask for this specific class
            if class_id not in class_masks:
                class_masks[class_id] = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            
            # Combine the current mask for this class
            class_masks[class_id] = np.maximum(class_masks[class_id], mask_resized)

        return class_masks

    @staticmethod
    def load_transformation(file_path):
        transform_1d = np.load(file_path)
        trans_mat = np.eye(4)
        rot = R.from_quat(transform_1d[3:])
        trans_mat[:3,:3] = rot.as_matrix()
        trans_mat[:3,3] = transform_1d[:3]
        return trans_mat



def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--msmap_yaml', 
                        type=str, 
                        help='YAML file of MS Map arguments')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    
    with open(args.msmap_yaml, 'r') as file:
        msmap_args = yaml.safe_load(file)
    
    ms_map = MS_Map(msmap_args)
    ms_map.make_map()