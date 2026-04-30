import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
from sklearn.cluster import HDBSCAN
import numpy as np
import open3d as o3d
import pickle as pkl

from utils import tensor_cosine_similarity, chunked_tensor_cosine_similarity
from terra_utils import TerraObject

from tqdm import tqdm

@torch.no_grad()
def compute_weighted_medoid(X, w, chunk_size=1024):
    best_score = float("-inf")
    best_idx = -1
    for start, scores in chunked_tensor_cosine_similarity(
        X,
        X,
        chunk_size=chunk_size
    ):
        sim = (scores * w.unsqueeze(0)).sum(dim=1)
        max_val, max_idx = torch.max(sim, dim=0)
        max_val = max_val.item()
        max_idx = max_idx.item()
        if max_val > best_score:
            best_score = max_val
            best_idx = start + max_idx
    return X[best_idx, :]
    # D = tensor_cosine_similarity(X, X)
    # total_similarity_score = (D * w.unsqueeze(0)).sum(dim=1)
    # medoid_idx = torch.argmax(total_similarity_score)
    # return X[medoid_idx,:]

def build_medoid_tensor(gidx_2_clipcounts, clip_segs, semantic_gidxs):
    # medoid_tensor = torch.zeros((len(semantic_gidxs), clip_segs.shape[1]), device=clip_segs.device)
    medoid_tensor = torch.zeros((len(semantic_gidxs), clip_segs.shape[1]), device='cpu', dtype=torch.float32)
    for idx, gidx in enumerate(tqdm(semantic_gidxs, desc="Building medoid tensor")):
        clipcounts_dict_entry = gidx_2_clipcounts.get(gidx, None)
        
        clip_ids = list(clipcounts_dict_entry.keys())
        X = torch.stack([clip_segs[clip_id,:] for clip_id in clip_ids]).to(clip_segs.device)
        w = torch.tensor(list(clipcounts_dict_entry.values()), device=clip_segs.device, dtype=torch.float32)
        medoid = compute_weighted_medoid(X, w)
        medoid_tensor[idx,:] = medoid.cpu()
    return medoid_tensor.to(clip_segs.device)

def compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.2):
    sorted_indices = torch.argsort(dist,dim=0) # smallest to largest dist from weighted mean
    num_keep = int(len(sorted_indices) * (1 - trim_percent))
    keep_idxs = sorted_indices[:num_keep]
    X_trimmed = X[keep_idxs]
    w_trimmed = w[keep_idxs]
    weighted_sum = (X_trimmed * w_trimmed.unsqueeze(1)).sum(dim=0)
    total_weight = w_trimmed.sum()
    return (weighted_sum / total_weight).squeeze()

def build_trimmed_mean_tensor(gidx_2_clipcounts, clip_segs, semantic_avg, semantic_gidxs, trim_percent=0.2):
    # trimmed_mean_tensor = torch.zeros((len(semantic_gidxs), clip_segs.shape[1]), device=clip_segs.device)
    trimmed_mean_tensor = torch.zeros((len(semantic_gidxs), clip_segs.shape[1]), device='cpu', dtype=torch.float32)
    for idx, gidx in enumerate(tqdm(semantic_gidxs, desc="Building trimmed mean tensor")):
        gidx2clipcounts_dict_entry = gidx_2_clipcounts.get(gidx, None)
        X = torch.stack([clip_segs[clip_id,:] for clip_id in gidx2clipcounts_dict_entry.keys()]).to(clip_segs.device)
        w = torch.tensor(list(gidx2clipcounts_dict_entry.values()), device=clip_segs.device, dtype=torch.float32)
        diff = tensor_cosine_similarity(X, semantic_avg[idx,:]) # [num_clip_ids,]
        dist = 1 - diff # convert cosine similarity to distance
        trimmed_mean = compute_weighted_trimmed_mean(X, w, dist, trim_percent)
        trimmed_mean_tensor[idx,:] = trimmed_mean.cpu()
    return trimmed_mean_tensor.to(clip_segs.device)

def weighted_median(x, w):
    idx = torch.argsort(x.squeeze())
    x_sorted = x[idx]
    w_sorted = w[idx]
    cdf = torch.cumsum(w_sorted, dim=0)
    if len(cdf.shape) == 0:
        return x_sorted
    return x_sorted[torch.searchsorted(cdf, 0.5)]

def compute_weighted_trimmed_medoid(X, w, z_mod_thresh=3.5):
    # Normalize embeddings
    X = X / X.norm(dim=1, keepdim=True)
    w_normed = w / w.sum()

    # Weighted mean in embedding space
    mu = (X * w_normed.unsqueeze(1)).sum(dim=0)
    mu = mu / mu.norm()

    # Cosine distance
    sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
    dist = 1 - sim
    
    # --- Compute z-scores ---
    # Robust stats
    med = weighted_median(dist, w_normed)
    mad = weighted_median(torch.abs(dist - med), w_normed) + 1e-8

    # Robust z-score
    z = 0.6745 * (dist - med) / mad
    
    inliers_mask = torch.abs(z) <= z_mod_thresh
    
    mu_trim = (X[inliers_mask[:,0],:] * w_normed[inliers_mask[:,0]].unsqueeze(1)).sum(dim=0)
    mu_trim = mu_trim / mu_trim.norm()
    return mu_trim

def build_trimmed_medoid_tensor(gidx_2_clipcounts, clip_segs, semantic_gidxs, z_mod_thresh=3.5):
    device_out = clip_segs.device  # final destination (GPU)
    # Move source once to CPU (or assume already CPU)
    clip_segs_cpu = clip_segs.cpu()
    trimmed_medoid_tensor = torch.zeros(
        (len(semantic_gidxs), clip_segs.shape[1]),
        device='cpu',
        dtype=torch.float32
    )
    for idx, gidx in enumerate(tqdm(semantic_gidxs, desc="Building trimmed medoid tensor")):
        clipcounts_dict_entry = gidx_2_clipcounts.get(gidx, None)
        if clipcounts_dict_entry is None:
            continue
        clip_ids = list(clipcounts_dict_entry.keys())

        # Stay on CPU
        X = torch.stack([clip_segs_cpu[clip_id, :] for clip_id in clip_ids])
        w = torch.tensor(list(clipcounts_dict_entry.values()), dtype=torch.float32)
        medoid = compute_weighted_trimmed_medoid(X, w, z_mod_thresh)
        trimmed_medoid_tensor[idx, :] = medoid  # already CPU

    # Move final result back to GPU once
    return trimmed_medoid_tensor.to(device_out)

def weighted_median_index(x, w):
    # Sort distances
    sorted_idx = torch.argsort(x[:,0])
    x_sorted = x[sorted_idx]
    w_sorted = w[sorted_idx]
    # CDF
    cdf = torch.cumsum(w_sorted, dim=0)
    # Find weighted median position
    median_pos = torch.searchsorted(cdf, 0.5)
    # Map back to original index
    return sorted_idx[median_pos]

def build_weighted_median_tensor(gidx_2_clipcounts, clip_segs, semantic_gidxs):
    device_out = clip_segs.device  # final destination (GPU)
    # Move source once to CPU (or assume already CPU)
    clip_segs_cpu = clip_segs.cpu()
    weighted_median_tensor = torch.zeros(
        (len(semantic_gidxs), clip_segs.shape[1]),
        device='cpu',
        dtype=torch.float32
    )
    for idx, gidx in enumerate(tqdm(semantic_gidxs, desc="Building weighted median tensor")):
        clipcounts_dict_entry = gidx_2_clipcounts.get(gidx, None)
        if clipcounts_dict_entry is None:
            continue
        clip_ids = list(clipcounts_dict_entry.keys())

        # Stay on CPU
        X = torch.stack([clip_segs_cpu[clip_id, :] for clip_id in clip_ids])
        w = torch.tensor(list(clipcounts_dict_entry.values()), dtype=torch.float32)
        
        # Normalize embeddings
        X = X / X.norm(dim=1, keepdim=True)
        w_normed = w / w.sum()
        # Weighted mean in embedding space
        mu = (X * w_normed.unsqueeze(1)).sum(dim=0)
        mu = mu / mu.norm()
        # Cosine distance
        sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
        dist = 1 - sim
        # --- Compute z-scores ---
        # Robust stats
        median_idx = weighted_median_index(dist, w_normed)
        weighted_median_tensor[idx, :] = X[median_idx, :]  # already CPU
        
        del X, w, dist, sim  # help memory

    # Move final result back to GPU once
    return weighted_median_tensor.to(device_out)

def compute_hdbscan_embedding(X, w, min_cluster_size=2, selection="weight", num_terrain=0):
    """
    X: [N, D] torch tensor
    w: [N] torch tensor
    returns: [D] torch tensor
    """
    # normalize for cosine geometry
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    wn = w / w.sum()
    Xw = Xn * wn.unsqueeze(1)
    Xw_np = Xw.detach().cpu().numpy()
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean"
    ).fit_predict(Xw_np)
    # group clusters
    clusters = {}
    for i, l in enumerate(labels):
        if l != -1:
            clusters.setdefault(l, []).append(i)

    # fallback → your strongest baseline
    if not clusters:
        return (X * w.unsqueeze(1)).sum(0) / w.sum()

    # pick cluster
    if selection == "size": # pick cluster with most elements
        best = max(clusters, key=lambda c: len(clusters[c]))
        idx = clusters[best]
        Xc, wc = X[idx], w[idx]
        out = (Xc * wc.unsqueeze(1)).sum(0) / wc.sum()
    elif selection == "average": # average across clusters
        out = torch.zeros(X.shape[1], device=X.device)
        counter = 0
        for i, (c, cluster_idxs) in enumerate(clusters.items()):
            cluster_X = X[cluster_idxs,:]
            cluster_w = w[cluster_idxs]
            cluster_embedding = (cluster_X * cluster_w.unsqueeze(1)).sum(0) / cluster_w.sum()
            cluster_embedding = cluster_embedding / cluster_embedding.norm()
            out += cluster_embedding * cluster_w.sum()
            counter += cluster_w.sum()
        out = out / counter
        out = out / out.norm()
    else:  # "weight" - pick cluster with largest total weight
        best = max(clusters, key=lambda c: wn[clusters[c]].sum().item())
        idx = clusters[best]
        Xc, wc = X[idx], w[idx]
        out = (Xc * wc.unsqueeze(1)).sum(0) / wc.sum()
    return out / (out.norm() + 1e-8)

def build_hdbscan_tensor(gidx_2_clipcounts, clip_segs, semantic_gidxs, 
                         min_cluster_size=2, selection="weight", num_terrain=0):
    """
    Returns:
        [num_gidxs, D] tensor
    """
    clip_segs_cpu = clip_segs.cpu()

    out = torch.zeros(
        (len(semantic_gidxs), clip_segs.shape[1]),
        dtype=torch.float32,
        device="cpu"
    )

    for i, gidx in enumerate(tqdm(semantic_gidxs, desc="HDBSCAN")):
        entry = gidx_2_clipcounts.get(gidx, None)
        if not entry:
            continue

        ids = list(entry.keys())

        X = torch.stack([clip_segs_cpu[j] for j in ids])
        w = torch.tensor(list(entry.values()), dtype=torch.float32)
        
        if X.shape[0] == 1:
            out[i,:] = X[0,:]
            continue

        out[i,:] = compute_hdbscan_embedding(
            X, w,
            min_cluster_size=min_cluster_size,
            selection=selection,
            num_terrain=num_terrain
        ).cpu()

    return out.to(clip_segs.device)


# def build_avgdist_tensor(gidx_2_clipcounts, clip_segs, semantic_gidxs):
#     # Load distance pickle file
#     outer_dir = "/ros2_bags/metric_data_0.5s/synced/sem_avg_fixed/"
#     with open(outer_dir + "provo_river_p1_11_3_25_cleaned/output/3_cam_dist/gidx2closestimgdist_dict_itr1829.pkl", "rb") as f:
#         gidx2dists = pkl.load(f)
#     # Precompute distance lookups
#     max_dist = max([dist for dist in gidx2dists.values()])
#     print("Max distance across all clip_ids:", max_dist,"[m]")
    
#     gidx2mean = {
#         gidx: abs(max_dist - np.mean(dists)) / max_dist for gidx, dists in gidx2dists.items()
#     }
    
#     device_out = clip_segs.device  # final destination (GPU)
#     # Move source once to CPU (or assume already CPU)    
#     clip_segs_cpu = clip_segs.cpu()
#     avgdist_tensor = torch.zeros(
#         (len(semantic_gidxs), clip_segs.shape[1]),
#         device='cpu',
#         dtype=torch.float32
#     )
#     for idx, gidx in enumerate(tqdm(semantic_gidxs, desc="Building avgdist tensor")):
#         clipcounts_dict_entry = gidx_2_clipcounts.get(gidx, None)
#         if clipcounts_dict_entry is None:
#             continue
#         clip_ids = torch.tensor(list(clipcounts_dict_entry.keys()), dtype=torch.long)
        
#         w_dist = gidx2mean[gidx]
        
#         X = clip_segs_cpu[clip_ids]
#         w = torch.tensor(list(clipcounts_dict_entry.values()), dtype=torch.float32)
#         w_normed = w / w.sum()
#         combined_w = w_normed
#         combined_w = combined_w / combined_w.sum()
#         weighted_avg_dist = (X * combined_w.unsqueeze(1)).sum(dim=0)
#         weighted_avg_dist = weighted_avg_dist / weighted_avg_dist.norm()
        
#         avgdist_tensor[idx, :] = w_dist * weighted_avg_dist.cpu()

#     return avgdist_tensor.to(device_out)


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
        elif method == "3dsg_avg":
            self._predict_3dsg(tasks_tensor, use_avg_clipids=True)
        elif method == "3dsg_max":
            self._predict_3dsg(tasks_tensor, use_avg_clipids=False)
        elif method == "aib":
            print("Not implemented. Should I?")
        else:
            print("Unrecognized object prediction method. Should be: [ms_avg, ms_max, 3dsg_avg, 3dsg_max, aib]")
            exit()
        print("Returning objects to Terra class")
        return self.objects
            
    def _predict_ms(self, tasks_tensor, use_avg_clipids, place_nodes_dict=None):
        if place_nodes_dict is None and use_avg_clipids: # use averaged clip_ids
            # ## MS-AVG
            # idx_scores = {}
            # matched_idxs = []
            # for start, scores in chunked_tensor_cosine_similarity(
            #     self.terra.semantic_gidx_avgclip,
            #     tasks_tensor,
            #     chunk_size=8192
            # ):
            #     max_scores, max_tasks = scores.max(dim=1)
            #     mask = (max_tasks >= self.terra.num_terrain) & (max_scores > self.terra.alpha)
            #     valid_idxs = mask.nonzero(as_tuple=True)[0]
            #     for local_idx in valid_idxs:
            #         idx_filt = start + local_idx.item()
            #         idx = self.terra.semantic_gidxs[idx_filt]
            #         # Move small data to CPU immediately
            #         idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
            #         matched_idxs.append(idx)
            #     # free GPU memory
            #     del scores
            
            ## MS-AVG-WITH-DISTANCE_WEIGHTS # TODO: implement...
            print("Running MS-Avg-Dist Method")
            # Load distance pickle file
            outer_dir = "/ros2_bags/metric_data_0.5s/synced/sem_avg_fixed/"
            with open(outer_dir + "provo_river_p1_11_3_25_cleaned/output/3_cam_dist/gidx2closestimgdist_dict_itr1829.pkl", "rb") as f:
                gidx2dist = pkl.load(f)
            # Precompute distance lookups
            max_dist = max([dist for dist in gidx2dist.values()])
            print("Max distance across all clip_ids:", max_dist,"[m]")
            gidx2weight = {
                gidx: abs(max_dist - dist) / max_dist for gidx, dist in gidx2dist.items()
            }          
            
            idx_scores = {}
            matched_idxs = []
            for start, scores in chunked_tensor_cosine_similarity(
                self.terra.semantic_gidx_avgclip,
                tasks_tensor,
                chunk_size=8192
            ):
                max_scores, max_tasks = scores.max(dim=1)
                mask = (max_tasks >= self.terra.num_terrain) & (max_scores > self.terra.alpha)
                valid_idxs = mask.nonzero(as_tuple=True)[0]
                for local_idx in valid_idxs:
                    idx_filt = start + local_idx.item()
                    idx = self.terra.semantic_gidxs[idx_filt]
                    # Move small data to CPU immediately
                    idx_scores[idx] = gidx2weight[idx] * scores[local_idx, self.terra.num_terrain:].cpu()
                    matched_idxs.append(idx)
                # free GPU memory
                del scores
            
            # ## MS-Medoid
            # print("Running MEDOID Method")
            # idx_scores = {}
            # matched_idxs = []
            # med_tensor = build_medoid_tensor(
            #     self.terra.gidx_2_clipcounts, 
            #     self.terra.clip_segs,
            #     self.terra.semantic_gidxs
            # )
            # print("Finished building medoid tensor")
            # for start, scores in chunked_tensor_cosine_similarity(
            #     med_tensor,
            #     tasks_tensor,
            #     chunk_size=8192
            # ):
            #     max_scores, max_tasks = scores.max(dim=1)
            #     mask = (max_tasks >= self.terra.num_terrain) & (max_scores > self.terra.alpha)
            #     valid_idxs = mask.nonzero(as_tuple=True)[0]
            #     for local_idx in valid_idxs:
            #         idx_filt = start + local_idx.item()
            #         idx = self.terra.semantic_gidxs[idx_filt]
            #         idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
            #         matched_idxs.append(idx)
            #     # free GPU memory
            #     del scores
            
            # ## MS-Trim
            # print("Running AVG-TRIMMED Method")
            # idx_scores = {}
            # matched_idxs = []
            # trim_percent = 0.6
            # print("Trim percent:",trim_percent)
            # avg_trimmed_tensor = build_trimmed_mean_tensor(
            #     self.terra.gidx_2_clipcounts, 
            #     self.terra.clip_segs,
            #     self.terra.semantic_gidx_avgclip,
            #     self.terra.semantic_gidxs,
            #     trim_percent=trim_percent
            # )
            # for start, scores in chunked_tensor_cosine_similarity(
            #     avg_trimmed_tensor,
            #     tasks_tensor,
            #     chunk_size=8192
            # ):
            #     max_scores, max_tasks = scores.max(dim=1)
            #     mask = (max_tasks >= self.terra.num_terrain) & (max_scores > self.terra.alpha)
            #     valid_idxs = mask.nonzero(as_tuple=True)[0]
            #     for local_idx in valid_idxs:
            #         idx_filt = start + local_idx.item()
            #         idx = self.terra.semantic_gidxs[idx_filt]
            #         idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
            #         matched_idxs.append(idx)
            #     # free GPU memory
            #     del scores
            
            
            # ## MS-Trim-Medoid
            # print("Running TRIMMED-MEDOID Method")
            # idx_scores = {}
            # matched_idxs = []
            # z_mod_thresh = 3.5
            # print("Z_mod-score threshold:",z_mod_thresh)
            # avg_trimmed_tensor = build_trimmed_medoid_tensor(
            #     self.terra.gidx_2_clipcounts, 
            #     self.terra.clip_segs,
            #     self.terra.semantic_gidxs,
            #     z_mod_thresh=z_mod_thresh
            # )
            # for start, scores in chunked_tensor_cosine_similarity(
            #     avg_trimmed_tensor,
            #     tasks_tensor,
            #     chunk_size=8192
            # ):
            #     max_scores, max_tasks = scores.max(dim=1)
            #     mask = (max_tasks >= self.terra.num_terrain) & (max_scores > self.terra.alpha)
            #     valid_idxs = mask.nonzero(as_tuple=True)[0]
            #     for local_idx in valid_idxs:
            #         idx_filt = start + local_idx.item()
            #         idx = self.terra.semantic_gidxs[idx_filt]
            #         idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
            #         matched_idxs.append(idx)
            #     # free GPU memory
            #     del scores
            
            # ## MS-Median
            # print("Running WEIGHTED-MEDIAN Method")
            # idx_scores = {}
            # matched_idxs = []
            # weighted_median_tensor = build_weighted_median_tensor(
            #     self.terra.gidx_2_clipcounts, 
            #     self.terra.clip_segs,
            #     self.terra.semantic_gidxs
            # )
            # for start, scores in chunked_tensor_cosine_similarity(
            #     weighted_median_tensor,
            #     tasks_tensor,
            #     chunk_size=8192
            # ):
            #     max_scores, max_tasks = scores.max(dim=1)
            #     mask = (max_tasks >= self.terra.num_terrain) & (max_scores > self.terra.alpha)
            #     valid_idxs = mask.nonzero(as_tuple=True)[0]
            #     for local_idx in valid_idxs:
            #         idx_filt = start + local_idx.item()
            #         idx = self.terra.semantic_gidxs[idx_filt]
            #         idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
            #         matched_idxs.append(idx)
            #     # free GPU memory
            #     del scores
            
            # ## MS-HDBSCAN
            # print("Running HDBSCAN Method")
            # idx_scores = {}
            # matched_idxs = []
            # selection_method = "average" # ["size","weight","average"]
            # print("Selection method:",selection_method)
            # hdb_tensor = build_hdbscan_tensor(
            #     self.terra.gidx_2_clipcounts, 
            #     self.terra.clip_segs,
            #     self.terra.semantic_gidxs,
            #     min_cluster_size=2,
            #     selection=selection_method,
            #     num_terrain=self.terra.num_terrain
            # )
            # print("Finished building hdbscan tensor")
            # for start, scores in chunked_tensor_cosine_similarity(
            #     hdb_tensor,
            #     tasks_tensor,
            #     chunk_size=8192
            # ):
            #     max_scores, max_tasks = scores.max(dim=1)
            #     mask = (max_tasks >= self.terra.num_terrain) & (max_scores > self.terra.alpha)
            #     valid_idxs = mask.nonzero(as_tuple=True)[0]
            #     for local_idx in valid_idxs:
            #         idx_filt = start + local_idx.item()
            #         idx = self.terra.semantic_gidxs[idx_filt]
            #         idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
            #         matched_idxs.append(idx)
            #     # free GPU memory
            #     del scores
            
        elif place_nodes_dict is None and not use_avg_clipids: # use max_clipid
            
            # Load distance pickle file
            outer_dir = "/ros2_bags/metric_data_0.5s/synced/sem_avg_fixed/"
            with open(outer_dir + "provo_river_p1_11_3_25_cleaned/output/3_cam_dist/gidx2closestimgdist_dict_itr1829.pkl", "rb") as f:
                gidx2dist = pkl.load(f)
            # Precompute distance lookups
            max_dist = max([dist for dist in gidx2dist.values()])
            print("Max distance across all clip_ids:", max_dist,"[m]")
            gidx2weight = {
                gidx: abs(max_dist - dist) / max_dist for gidx, dist in gidx2dist.items()
            }   
            
            clipid_scores = {}
            for start, scores in chunked_tensor_cosine_similarity(
                self.terra.clip_segs,
                tasks_tensor,
                chunk_size=8192
            ):
                max_scores, max_tasks = scores.max(dim=1)
                for local_clip_idx in range(scores.shape[0]):
                    clip_id = start + local_clip_idx
                    if max_tasks[local_clip_idx] >= self.terra.num_terrain and max_scores[local_clip_idx] > self.terra.alpha:
                        clipid_scores[clip_id] = scores[local_clip_idx, self.terra.num_terrain:].cpu()
                del scores
            idx_scores = {}
            matched_idxs = []
            for idx, clip_ids in self.terra.gidx_2_clipcounts.items():
                curr_clipid, curr_count = max(self.terra.gidx_2_clipcounts[idx].items(), key=lambda x: x[1])
                if curr_clipid not in clipid_scores:
                    continue
                # idx_scores[idx] = (1 + gidx2weight[idx]) * clipid_scores[curr_clipid]
                idx_scores[idx] = gidx2weight[idx] * clipid_scores[curr_clipid]
                matched_idxs.append(idx)
                
        elif use_avg_clipids: # use average clipids with place node filtering
            task_pts = {}
            for task_idx, place_nodes in place_nodes_dict.items():
                if len(place_nodes) == 0:
                    task_pts[task_idx] = set()
                    continue    
                place_pos = np.stack([self.terra.terra_3dsg.nodes[n]["pos"] for n in place_nodes])
                idxs_list = self.kdt_2d.query_ball_point(place_pos, r=10)
                task_pts[task_idx] = set(np.concatenate(idxs_list))
            idx_scores = {}
            matched_idxs = set()
            for start, scores in chunked_tensor_cosine_similarity(
                self.terra.semantic_gidx_avgclip,
                tasks_tensor,
                chunk_size=8192
            ): # (chunk_size, num_terrain+num_tasks)
                for task_idx in range(self.terra.num_terrain, scores.shape[1]):
                    local_idxs = (scores[:, task_idx] > self.terra.alpha).nonzero(as_tuple=True)[0]
                    for local_idx in local_idxs:
                        idx_filt = start + local_idx.item()
                        idx = self.terra.semantic_gidxs[idx_filt]
                        max_score = scores[local_idx,:].max().item()
                        max_task = scores[local_idx,:].argmax().item()
                        if (
                            max_task >= self.terra.num_terrain
                            and max_score > self.terra.alpha
                            and idx in task_pts[task_idx - self.terra.num_terrain]
                        ):
                            idx_scores[idx] = scores[local_idx, self.terra.num_terrain:].cpu()
                            matched_idxs.add(idx)
                del scores
            
        else: # use max_clipids with place node filtering
            task_pts = {}
            for task_idx, place_nodes in place_nodes_dict.items():
                if len(place_nodes) == 0:
                    task_pts[task_idx] = set()
                    continue  
                place_pos = np.stack([self.terra.terra_3dsg.nodes[n]["pos"] for n in place_nodes])
                idxs_list = self.kdt_2d.query_ball_point(place_pos, r=10)
                task_pts[task_idx] = set(np.concatenate(idxs_list))
            clipid_scores = {}
            for start, scores in chunked_tensor_cosine_similarity(
                self.terra.clip_segs,
                tasks_tensor,
                chunk_size=8192
            ):
                max_scores, max_tasks = scores.max(dim=1)
                for local_clip_idx in range(scores.shape[0]):
                    clip_id = start + local_clip_idx
                    if max_tasks[local_clip_idx] >= self.terra.num_terrain and max_scores[local_clip_idx] > self.terra.alpha:
                        clipid_scores[clip_id] = scores[local_clip_idx, self.terra.num_terrain:].cpu()
                del scores
                
            idx_scores = {}
            matched_idxs = []
            for idx, clip_ids in self.terra.gidx_2_clipcounts.items():
                curr_clipid, curr_count = max(self.terra.gidx_2_clipcounts[idx].items(), key=lambda x: x[1])
                if curr_clipid not in clipid_scores:
                    continue
                
                task_match = False
                for task_idx in task_pts.keys():
                    if idx in task_pts[task_idx]:
                        task_match = True
                        break
                if not task_match:
                    continue
                idx_scores[idx] = clipid_scores[curr_clipid]
                matched_idxs.append(idx)
        
        # Cluster object points & extract bounding boxes from clusters
        if len(matched_idxs) == 0:
            print("No similar MS-Map indices detected")
            exit()
        print("Made it to start clustering points into bboxes")
        self._cluster_into_bboxes(list(matched_idxs), idx_scores)
        print("Finished clustering points into bboxes")
        
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
    
    def _predict_3dsg(self, tasks_tensor, use_avg_clipids):
        chosen_region_nodes = self._predict_object_regions(tasks_tensor)
        chosen_place_nodes = self._predict_object_places(tasks_tensor, region_nodes_dict=chosen_region_nodes)
        self._predict_ms(tasks_tensor, use_avg_clipids=use_avg_clipids, place_nodes_dict=chosen_place_nodes)
    
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
                    
                    bottom_region_node = False
                    for nbr in self.terra.terra_3dsg.neighbors(node):
                        nbr_level = self.terra.terra_3dsg.nodes[nbr]["level"]
                        if nbr_level == 1: # children are places, stop descent
                            bottom_region_node = True
                            break            
                    if bottom_region_node:
                        selected.add(node)
                        continue
                    
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