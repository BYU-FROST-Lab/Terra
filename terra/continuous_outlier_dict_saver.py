from argparse import ArgumentParser
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re

import hdbscan
from sklearn.metrics import silhouette_score

from utils import tensor_cosine_similarity, chunked_tensor_cosine_similarity

def cosine_geometric_median(X, w=None, eps=1e-5, max_iter=50):
    """
    Cosine-based geometric median for L2-normalized embeddings.

    X: (N, D)
    w: optional (N,)
    """

    # Normalize (critical for cosine geometry)
    X = X / X.norm(dim=1, keepdim=True)

    if w is not None:
        w = w / w.sum()

    # initialize with weighted mean direction (good start)
    if w is None:
        y = X.mean(dim=0)
    else:
        y = (X * w.unsqueeze(1)).sum(dim=0)

    y = y / y.norm()

    for _ in range(max_iter):
        # cosine distance surrogate: 1 - dot
        sim = X @ y
        dist = (1 - sim).clamp(min=eps)

        # weights: closer points matter more
        if w is None:
            weights = 1.0 / dist
        else:
            weights = w / dist

        y_new = (weights.unsqueeze(1) * X).sum(dim=0)
        y_new = y_new / y_new.norm()

        if torch.norm(y - y_new) < eps:
            break

        y = y_new

    return y


def count_outliers_geomedian(X, w=None, std_dev=3):
    N = X.shape[0]

    # Normalize embeddings
    X = X / X.norm(dim=1, keepdim=True)

    # ---- Compute robust center ----
    # mu = geometric_median(X, w)
    mu = cosine_geometric_median(X, w)

    # ---- Cosine distance ----
    sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
    dist = 1 - sim

    # ---- FAST PATH: no weights or uniform weights ----
    if w is None or torch.all(w == 1):
        med = dist.median()
        mad = torch.median(torch.abs(dist - med)) + 1e-8

        z = 0.6745 * (dist - med) / mad
        outliers = torch.abs(z) > std_dev

        return outliers.sum()

    # ---- Weighted case ----
    w = w / w.sum()

    med = weighted_median(dist, w)
    mad = weighted_median(torch.abs(dist - med), w) + 1e-8

    z = 0.6745 * (dist - med) / mad
    outliers = torch.abs(z) > std_dev

    return (w * outliers.float()).sum()


def weighted_median(x, w=None):
    x = x.squeeze()

    # Unweighted case (fast path)
    if w is None:
        return x.median()

    w = w.squeeze()

    idx = torch.argsort(x)
    x_sorted = x[idx]
    w_sorted = w[idx]

    cdf = torch.cumsum(w_sorted, dim=0)
    cutoff = 0.5 * w_sorted.sum()

    return x_sorted[torch.searchsorted(cdf, cutoff)]


def count_outliers_meanmedian(X, w=None, std_dev=3):
    N = X.shape[0]

    # Normalize embeddings
    X = X / X.norm(dim=1, keepdim=True)

    # ---- FAST PATH: no weights or uniform weights ----
    if w is None or torch.all(w == 1):
        mu = X.mean(dim=0)
        mu = mu / mu.norm()

        sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
        dist = 1 - sim

        med = dist.median()
        mad = torch.median(torch.abs(dist - med))

        z = 0.6745 * (dist - med) / mad
        outliers = torch.abs(z) > std_dev

        return outliers.sum()

    # ---- Weighted case (only if actually needed) ----
    w = w / w.sum()

    mu = (X * w.unsqueeze(1)).sum(dim=0)
    mu = mu / mu.norm()

    sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
    dist = 1 - sim

    med = weighted_median(dist, w)
    mad = weighted_median(torch.abs(dist - med), w) + 1e-8

    z = 0.6745 * (dist - med) / mad
    outliers = torch.abs(z) > std_dev

    return (w * outliers.float()).sum()

# def count_outliers(X, w, threshold=0.3):
#     # Normalize embeddings
#     X = X / X.norm(dim=1, keepdim=True)
#     w = w / w.sum()

#     # Weighted mean in embedding space
#     mu = (X * w.unsqueeze(1)).sum(dim=0)# / w.sum()
#     mu = mu / mu.norm()

#     # Cosine distance
#     sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
#     dist = 1 - sim

#     # --- Outlier mask ---
#     outliers = dist > threshold

#     # Count outliers
#     return outliers.count_nonzero()

def compute_distances_from_mean(X, w=None):
    N = X.shape[0]

    # Normalize embeddings
    X = X / X.norm(dim=1, keepdim=True)

    # Compute center (weighted mean)
    if w is None:
        mu = X.mean(dim=0)
    else:
        w = w / w.sum()
        mu = (X * w.unsqueeze(1)).sum(dim=0)

    mu = mu / mu.norm()

    # Cosine distance
    sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
    dist = 1 - sim
    return dist.cpu().detach().numpy()
    # return dist.mean().item()

def compute_distances_from_median(X, w=None):
    N = X.shape[0]

    # Normalize embeddings
    X = X / X.norm(dim=1, keepdim=True)

    # Compute center (weighted mean)
    if w is None:
        mu = X.mean(dim=0)
    else:
        w = w / w.sum()
        mu = (X * w.unsqueeze(1)).sum(dim=0)

    mu = mu / mu.norm()

    # Cosine distance
    sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
    dist = 1 - sim
    
    med = dist.median()        
    dists_from_med = torch.abs(dist - med)
    return dists_from_med.cpu().detach().numpy()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to load the saved msmap output files.")
    parser.add_argument('--itr', type=int, required=True, help="Iteration of saved dictionaries.")
    args = parser.parse_args()
    
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    itr = args.itr
        
    output_dir = Path(args.output_dir)
    with open(output_dir / f"clip_segs_itr{itr}.pt", "rb") as f:
        clip_ids = torch.load(f)
    with open(output_dir / f"gidx2clipcounts_dict_itr{itr}.pkl", "rb") as f:
        gidx_2_clipcounts = pkl.load(f)
    with open(output_dir / f"gidx2clipdists_dict_itr{itr}.pkl", "rb") as f:
        gidx_2_clipdists = pkl.load(f)
    with open(output_dir / f"gidx2clipmaskidx_dict_itr{itr}.pkl", "rb") as f:
        gidx_2_clipmaskidx = pkl.load(f)
    with open(output_dir / f"saved_fastsam_mask_names_itr{itr}.pkl", "rb") as f:
        fsam_mask_names = pkl.load(f)
    
    max_outliers = -1
    chosen_gidx = None
    clipid_2_counts = None
    
    gidx_2_outlier_counts = {}
    gidx_2_total_counts = {}
    gidx_2_dists_from_mean = {}
    gidx_2_dists_from_med = {}
    gidx_2_dists_from_lidar = {}
    gidx_2_cluster_counts = {}
    gidx_2_obsv_counts = {}
    gidx_2_silhouette_scores = {}
    gidx_2_dbcv_scores = {}
    # count_cutoff = 200
    # count_idx = 0
    gidx_list = []
    hdb = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=2,
        min_samples=2, #2
    )
    # hdb = hdbscan.HDBSCAN(
    #     metric='euclidean',
    #     min_cluster_size=2,
    #     min_samples=2, #2
    # )
    err_cnt = 0
    for g_idx, cid_2_cnt in tqdm(gidx_2_clipcounts.items(),desc="Selecting point with most outliers"):        
        ## Remove terrain text embeddings
        cid_2_cnt_old = dict(cid_2_cnt)
        for i, (clip_id, count) in enumerate(cid_2_cnt_old.items()):
            if clip_id < 7: # Assuming terrain clip IDs are 0-6, adjust if needed
                del cid_2_cnt[clip_id]
                del gidx_2_clipdists[g_idx][clip_id]
                del gidx_2_clipmaskidx[g_idx][clip_id]
        
        gidx_list.append(g_idx)
        
        # Build X
        X = torch.zeros((len(cid_2_cnt), 512), device=device)
        dists = []
        for i, (clip_id, count) in enumerate(cid_2_cnt.items()):
            # if clip_id < terra.num_terrain:
            #     print(f"Error: clip_id {clip_id} is less than num_terrain {terra.num_terrain}")
            X[i,:] = clip_ids[clip_id,:]
            dists.append(gidx_2_clipdists[g_idx][clip_id])
            
        # Count number of clusters
        if X.shape[0] > 1:
            # semantic_dists = 1 - tensor_cosine_similarity(X,X)
            # labels = hdb.fit_predict(semantic_dists.detach().cpu().numpy().astype(np.float64))
            # del semantic_dists
            
            semantic_dists = np.zeros((X.shape[0],X.shape[0]),dtype=np.float64)
            for start, scores in chunked_tensor_cosine_similarity(X,X,chunk_size=128): # (chunk_size, x.shape[0])
                scores_cpu = scores.detach().cpu().numpy().astype(np.float64)
                num_rows_in_chunk = scores_cpu.shape[0]
                semantic_dists[start : start + num_rows_in_chunk, :] = 1 - scores_cpu
                del scores    
            np.fill_diagonal(semantic_dists, 0)
            semantic_dists = np.clip(semantic_dists, a_min=0.0, a_max=1.0)
            labels = hdb.fit_predict(semantic_dists)
            
            ## Note: DBCV not any more informative than silhuoette score b/c 
            ## CLIP embeddings are not twisted but ellipsoidal in shape which silhuoette works just fine on
            # try:
            #     dbcv_score = hdbscan.validity.validity_index(
            #         semantic_dists,
            #         labels,
            #         metric='precomputed',
            #         d=512, # number of features
            #     )
            #     gidx_2_dbcv_scores[g_idx] = dbcv_score
            # except (AssertionError, ValueError, ZeroDivisionError):
            #     err_cnt += 1
            #     # print(f"\nERROR {err_cnt}\n")
                
            # labels = hdb.fit_predict(X.detach().cpu().numpy())
            
            if X.shape[0] > 2:
                valid_mask = labels != -1
                if np.count_nonzero(valid_mask) > 2:
                    s_score = silhouette_score(
                        semantic_dists[valid_mask][:, valid_mask],
                        labels[valid_mask],
                        metric='precomputed'
                    )
                    gidx_2_silhouette_scores[g_idx] = s_score
            
            num_clusters = np.unique(labels).shape[0]
            gidx_2_cluster_counts[g_idx] = num_clusters
        else:
            gidx_2_cluster_counts[g_idx] = 1
        
        gidx_2_obsv_counts[g_idx] = X.shape[0]
            
        gidx_2_dists_from_mean[g_idx] = compute_distances_from_mean(X, w=None)
        gidx_2_dists_from_med[g_idx] = compute_distances_from_median(X, w=None)
        gidx_2_dists_from_lidar[g_idx] = dists
        
        num_outliers = count_outliers_meanmedian(X, std_dev=3.5)
        # num_outliers = count_outliers_geomedian(X, std_dev=3.5)
        # num_outliers = count_outliers(X, w, threshold=0.3)

        gidx_2_outlier_counts[g_idx] = num_outliers.detach().cpu().item()
        gidx_2_total_counts[g_idx] = len(cid_2_cnt) if len(cid_2_cnt) > 0 else 1
        
        if num_outliers > max_outliers:
            max_outliers = num_outliers
            chosen_gidx = g_idx
            clipid_2_counts = cid_2_cnt
            print("New max number of outliers",max_outliers)

    # Compute Global Outlier Ratio
    num = 0.0
    den = 0.0
    min_ratio = 1.0
    max_ratio = 0.0
    gidx_2_outlier_ratio = {}
    for g_idx in gidx_2_outlier_counts:
        num += gidx_2_outlier_counts[g_idx]
        den += gidx_2_total_counts[g_idx]
        ratio = gidx_2_outlier_counts[g_idx] / gidx_2_total_counts[g_idx]
        if ratio < min_ratio:
            min_ratio = ratio
        if ratio > max_ratio:
            max_ratio = ratio        
        gidx_2_outlier_ratio[g_idx] = ratio
        
    # Save dictionaries
    with open(output_dir / f"gidx2outliercounts_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_outlier_counts, f)
    with open(output_dir / f"gidx2counts_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_total_counts, f)
    with open(output_dir / f"gidx2obsvmediandists_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_dists_from_med, f)
    with open(output_dir / f"gidx2obsvmeandists_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_dists_from_mean, f)
    with open(output_dir / f"gidx2obsvlidardists_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_dists_from_lidar, f)
    with open(output_dir / f"gidx2clustercounts_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_cluster_counts, f)
    with open(output_dir / f"gidx2obsvcounts_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_obsv_counts, f)
    with open(output_dir / f"gidx2outlierratios_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_outlier_ratio, f)
    with open(output_dir / f"gidx2silscores_noterrain_dict_itr{itr}.pkl", "wb") as f:
        pkl.dump(gidx_2_silhouette_scores, f)
    # with open(output_dir / f"gidx2dbcv_noterrain_dict_itr{itr}.pkl", "wb") as f:
    #     pkl.dump(gidx_2_dbcv_scores, f)
    