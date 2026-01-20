import torch
from scipy.spatial import KDTree
# from sklearn.cluster import DBSCAN
import numpy as np
import open3d as o3d

from utils import tensor_cosine_similarity, chunked_tensor_cosine_similarity
from terra_utils import TerraObject

class ObjectPredictor:
    """
    Responsible for predicting Terra Objects according to task
    """
    
    def __init__(self, terra, bounds = [-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf]):
        self.terra = terra
        self.kdt_2d = KDTree(terra.pc[:,:2])
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax = bounds
        
        self.objects = []
    
    def predict(self, tasks_tensor, method="ms_avg"):        
        self.objects = []
        print("Method for object retrieval:",method)
        if method == "ms_avg":
            self._predict_ms(tasks_tensor, use_avg_clipids=True)
        elif method == "ms_max":
            self._predict_ms(tasks_tensor, use_avg_clipids=False)
        elif method == "3dsg":
            self._predict_3dsg(tasks_tensor)
        elif method == "aib":
            print("Not implemented. Should I?")
        else:
            print("Unrecognized object prediction method. Should be: [ms_avg, ms_max, 3dsg, aib]")
            exit()
        return self.objects
            
    def _predict_ms(self, tasks_tensor, use_avg_clipids, place_nodes_dict=None):
        if place_nodes_dict is None and use_avg_clipids: # use averaged clip_ids
            idx_scores = {}
            matched_idxs = []
            for start, scores in chunked_tensor_cosine_similarity(
                self.terra.clip_tensor_semanticpc,
                tasks_tensor,
                chunk_size=8192
            ):
                max_scores, max_tasks = scores.max(dim=1)
                mask = (max_tasks >= self.terra.num_terrain) & (max_scores > self.terra.alpha)
                valid_idxs = mask.nonzero(as_tuple=True)[0]
                for local_idx in valid_idxs:
                    idx_filt = start + local_idx.item()
                    idx = self.terra.semantic_pc_idxs[idx_filt]
                    # Move small data to CPU immediately
                    idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
                    matched_idxs.append(idx)
                # free GPU memory
                del scores
            
        elif place_nodes_dict is None and not use_avg_clipids: # use max_clipid
            idx_scores = {}
            matched_idxs = []
            for start, scores in chunked_tensor_cosine_similarity(
                self.terra.clip_tensor,
                tasks_tensor,
                chunk_size=8192
            ):
                max_scores, max_tasks = scores.max(dim=1)
                for local_idx in range(scores.shape[0]):
                    clip_id = start + local_idx
                    if clip_id not in self.terra.pcidx_2_clipid:
                        continue
                    if max_tasks[local_idx] >= self.terra.num_terrain and max_scores[local_idx] > self.terra.alpha:
                        idx_scores[clip_id] = scores[local_idx, self.terra.num_terrain:].cpu()
                        matched_idxs.append(clip_id)
                del scores
                
        else:
            task_pts = {}
            for task_idx, place_nodes in place_nodes_dict.items():
                place_pos = np.stack([self.terra.terra_3dsg.nodes[n]["pos"] for n in place_nodes])
                idxs_list = self.kdt_2d.query_ball_point(place_pos, r=10)
                task_pts[task_idx] = set(np.concatenate(idxs_list))
            idx_scores = {}
            matched_idxs = set()
            for start, scores in chunked_tensor_cosine_similarity(
                self.terra.clip_tensor_semanticpc,
                tasks_tensor,
                chunk_size=8192
            ):
                for task_idx in range(self.terra.num_terrain, scores.shape[1]):
                    local_idxs = (scores[:, task_idx] > self.terra.alpha).nonzero(as_tuple=True)[0]
                    for local_idx in local_idxs:
                        idx_filt = start + local_idx.item()
                        idx = self.terra.semantic_pc_idxs[idx_filt]
                        max_score = scores[local_idx].max().item()
                        max_task = scores[local_idx].argmax().item()
                        if (
                            max_task >= self.terra.num_terrain
                            and max_score > self.terra.alpha
                            and idx in task_pts[task_idx - self.terra.num_terrain]
                        ):
                            idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
                            matched_idxs.add(idx)
                del scores
        
        # Cluster object points & extract bounding boxes from clusters
        if len(matched_idxs) == 0:
            print("No similar MS-Map indices detected")
            exit()
        self._cluster_into_bboxes(list(matched_idxs), idx_scores)
        
    def _cluster_into_bboxes(self, matched_idxs, idx_scores):
        surrounding_point_indices = self.terra.kdt.query_ball_point(
            self.terra.pc[matched_idxs,:], self.terra.search_rad)
        indices = np.unique(np.concatenate(surrounding_point_indices)).astype(int)
        surrounding_points = self.terra.pc[indices,:]
        
        clustered_points = self.terra.dbscan.fit_predict(surrounding_points)
        unique_cluster_ids, counts = np.unique(clustered_points[clustered_points != -1], return_counts=True)
        for cluster_id, counts in zip(unique_cluster_ids, counts):
            if counts < 4:
                continue
            cluster_pt_indices = np.where(clustered_points == cluster_id)[0]
            clustered_surrounding_points = surrounding_points[cluster_pt_indices]
            
            # Avg across points in most consistent task
            cluster_tasks = {}
            cluster_task_max_score = {}
            cluster_task_scores = {}
            for cluster_pt_idx in cluster_pt_indices:
                idx = indices[cluster_pt_idx]
                if idx in idx_scores:
                    idx_task = idx_scores[idx].argmax().item()
                    idx_max_score = idx_scores[idx].max().item()
                    if idx_task in cluster_tasks:
                        cluster_tasks[idx_task] += 1
                        cluster_task_max_score[idx_task] = cluster_task_max_score[idx_task] + idx_max_score
                        cluster_task_scores[idx_task] = cluster_task_scores[idx_task] + idx_scores[idx]
                        # cluster_task_max_score[idx_task] += idx_max_score
                        # cluster_task_scores[idx_task] += idx_scores[idx]
                    else:
                        cluster_tasks[idx_task] = 1
                        cluster_task_max_score[idx_task] = idx_max_score
                        cluster_task_scores[idx_task] = idx_scores[idx]
            if len(cluster_tasks) == 0:
                continue
            
            max_cluster_task = max(cluster_tasks, key=lambda k: cluster_tasks[k])
            max_cluster_score = cluster_task_max_score[max_cluster_task] / cluster_tasks[max_cluster_task]
            cluster_scores = cluster_task_scores[max_cluster_task] / cluster_tasks[max_cluster_task]
            if max_cluster_score == 0.0:
                continue
            
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(clustered_surrounding_points)
            bb_color = [0,0,1]#distinct_colors[l] if l < len(distinct_colors) else random_color()
            cluster_pcd.paint_uniform_color(bb_color)#[0.5,0.5,0.5])#grays[3])
            obb = cluster_pcd.get_oriented_bounding_box() # OR get_axis_aligned_bounding_box()
            obb.color = bb_color
            
            if self._is_obb_in_bounds(obb):
                self.objects.append(TerraObject(cluster_scores, obb, self.terra.prev_task_idx))
    
    def _is_obb_in_bounds(self, obb):
        cx,cy,cz = obb.center # LIOSAM frame (note: bounds in WORLD frame)
        T = np.eye(4) # TODO: Make this an argument? get_liosam2orig_transformation(self.experiment_type) # transformation from LIOSAM to WORLD
        P_L = np.array([[cx,cy,cz,1.0]]).T # homogeneous coords
        P_W = T @ P_L
        cx_w, cy_w, cz_w = P_W[0], P_W[1], P_W[2]
        if (self.xmin <= cx_w <= self.xmax) and (self.ymin <= cy_w <= self.ymax) and (self.zmin <= cz_w <= self.zmax):
            return True
        else:
            return False
    
    def _predict_3dsg(self, tasks_tensor):
        chosen_region_nodes = self._predict_object_regions(tasks_tensor)
        chosen_place_nodes = self._predict_object_places(tasks_tensor, region_nodes_dict=chosen_region_nodes)
        self._predict_ms(tasks_tensor, use_avg_clipids=True, place_nodes_dict=chosen_place_nodes)
    
    def _predict_object_regions(self, tasks_tensor):
        region_nodes = [n for n, d in self.terra.terra_3dsg.nodes(data=True) if d["level"] > 1]
        nodeid_to_idx = {n: i for i, n in enumerate(region_nodes)}
        region_embeddings = torch.vstack([self.terra.terra_3dsg.nodes[n]["embedding"] for n in region_nodes])
        scores = tensor_cosine_similarity(
            region_embeddings, 
            tasks_tensor[self.terra.num_terrain:,:]) # (num_region_nodes, num_tasks)
        region_node_dict = self.terra.nodes_above_level(min_level=1) # {node_level: [n_id1, ...], ...}
        start_regions = {}
        for task_idx in range(scores.shape[1]):
            max_score = scores[:, task_idx].max().item()
            n_idx = scores[:, task_idx].argmax().item()
            best_layer = self.terra.terra_3dsg.nodes[region_nodes[n_idx]]["level"]
            
            nodes_in_layer = region_node_dict[best_layer]
            
            start_regions[task_idx] = [
                n for n in nodes_in_layer
                if scores[nodeid_to_idx[n], task_idx] > self.terra.alpha
            ]
            if len(start_regions[task_idx]) == 0:
                # print("No regions above cos-sim thresh")
                start_regions[task_idx] = [n for n in nodes_in_layer]
        
            # Descend through the tree to collect all child nodes
            selected_regions = {}
            for task_idx, starting_nodes in start_regions.items():
                selected = set()
                queue = list(starting_nodes)
                while queue:
                    node = queue.pop()
                    node_level = self.terra.terra_3dsg.nodes[node]["level"]
                    if node_level == 2:
                        selected.add(node)
                    # Explore children until reaching level 2
                    for nbr in self.terra.terra_3dsg.neighbors(node):
                        nbr_level = self.terra.terra_3dsg.nodes[nbr]["level"]
                        if 2 <= nbr_level < node_level:
                            queue.append(nbr)
                selected_regions[task_idx] = selected    
        
        return selected_regions
    
    def _predict_object_places(self, tasks_tensor, region_nodes_dict=None):
        start_places = {}
        if region_nodes_dict is not None:
            for task_idx, region_nodes in region_nodes_dict.items():
                start_places[task_idx] = set()
                for rn in region_nodes:
                    for nbr in self.terra.terra_3dsg.neighbors(rn):
                        nbr_level = self.terra.terra_3dsg.nodes[nbr]["level"]
                        if nbr_level == 1: # places layer
                            start_places[task_idx].add(nbr)
            
        place_nodes = [n for n, d in self.terra.terra_3dsg.nodes(data=True) if d["level"] == 1]
        nodeid_to_idx = {n: i for i, n in enumerate(place_nodes)}
        place_embeddings = torch.vstack([self.terra.terra_3dsg.nodes[n]["embedding"] for n in place_nodes])
        scores = tensor_cosine_similarity(
            place_embeddings, 
            tasks_tensor[self.terra.num_terrain:,:]) # (num_region_nodes, num_tasks)
        selected_places = {}
        for task_idx in range(scores.shape[1]):                
            if region_nodes_dict is None:
                selected_places[task_idx] = [
                    n for n in place_nodes
                    if scores[nodeid_to_idx[n], task_idx] > self.terra.alpha
                ]
            else:
                selected_places[task_idx] = [
                    n for n in place_nodes
                    if (scores[nodeid_to_idx[n], task_idx] > self.terra.alpha)
                    and n in start_places[task_idx]
                ]
            if len(selected_places[task_idx]) == 0:
                selected_places[task_idx] = start_places[task_idx]
        
        return selected_places