from argparse import ArgumentParser
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

import open3d as o3d

from terra_utils import load_terra
from utils import tensor_cosine_similarity

def compute_weighted_medoid(X, w):
    D = tensor_cosine_similarity(X, X)
    total_similarity_score = (D * w.unsqueeze(0)).sum(dim=1)
    medoid_idx = torch.argmax(total_similarity_score)
    return X[medoid_idx,:]

def compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.2):
    sorted_indices = torch.argsort(dist,dim=0) # smallest to largest dist from weighted mean
    num_keep = int(len(sorted_indices) * (1 - trim_percent))
    print(num_keep)
    keep_idxs = sorted_indices[:num_keep]
    X_trimmed = X[keep_idxs]
    w_trimmed = w[keep_idxs]
    weighted_sum = (X_trimmed * w_trimmed.unsqueeze(1)).sum(dim=0)
    total_weight = w_trimmed.sum()
    return (weighted_sum / total_weight).squeeze()

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
    
    

# def compute_weighted_geometric_median(X, w, max_iter=100, tol=1e-6):
#     # Initialize with weighted mean
#     best_x = (X * w.unsqueeze(1)).sum(dim=0) / w.sum()
#     for i in range(max_iter):
#         # Compute distances from current point to all points
#         # diff = X - best_x.unsqueeze(0)
#         # dists = diff.norm(dim=1).clamp(min=1e-12)
#         diff = tensor_cosine_similarity(X, best_x.unsqueeze(0)) # [num_clip_ids,]
#         dist = 1 - diff # convert cosine similarity to distance
        
#         # Compute weights inversely proportional to distances
#         weights = w / (dist.squeeze() + 1e-12) # avoid division by zero
        
#         # Update current point using weighted average of all points with new weights
#         new_x = (X * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

#         if (new_x - best_x).norm() < tol:
#             print("Converged after {} iterations".format(i+1))
#             break
#         best_x = new_x
#     return best_x

from sklearn.cluster import HDBSCAN
def compute_hdbscan_embedding(X, w, min_cluster_size=2, selection="weight"):
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
    if selection == "size":
        best = max(clusters, key=lambda c: len(clusters[c]))
    else:  # "weight"
        best = max(clusters, key=lambda c: wn[clusters[c]].sum().item())
    idx = clusters[best]
    Xc, wc = X[idx], w[idx]
    out = (Xc * wc.unsqueeze(1)).sum(0) / wc.sum()
    return out / (out.norm() + 1e-8)

def weighted_median(x, w):
    idx = torch.argsort(x.squeeze())
    x_sorted = x[idx]
    w_sorted = w[idx]
    cdf = torch.cumsum(w_sorted, dim=0)
    if len(cdf.shape) == 0:
        return x_sorted
    return x_sorted[torch.searchsorted(cdf, 0.5)]

def count_outliers(X, w, std_dev=3):
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
    
    outliers = torch.abs(z) > std_dev

    # # --- Weighted variance ---
    # var_d = (w * dist**2).sum() / (X.shape[0] - 1) # sample variance

    # # --- Standard deviation ---
    # std_d = torch.sqrt(var_d + 1e-8)
    
    # # --- Outlier threshold ---
    # threshold = std_dev * std_d

    # # --- Outlier mask ---
    # outliers = dist > threshold

    # Count outliers
    # num_outliers = outliers.count_nonzero()
    num_outliers = (w * (outliers.squeeze())).sum()
    
    return num_outliers

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

def jet_colormap(t):
    # t in [0,1]
    r = np.clip(1.5 - np.abs(4*t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*t - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--Terra', type=str, help="Terra class object filepath")
    parser.add_argument('--terrain_gidx', type=str, help="List of global indices assigned to terrain")
    # parser.add_argument('--num_imgs', type=int, default=-1, help="Number of images to process (-1 = all)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    # Load Terra
    terra = load_terra(args.Terra)
    gidx_2_clipcounts = terra.gidx_2_clipcounts
    clip_ids = terra.clip_segs # [num_clip_ids, 512]
    
    with open(args.terrain_gidx, "rb") as f:
        terrain_gidxs = pkl.load(f)
    
    max_outliers = -1
    chosen_gidx = None
    clipid_2_counts = None
    
    gidx_2_outlier_counts = {}
    gidx_2_total_counts = {}
    for g_idx, cid_2_cnt in tqdm(gidx_2_clipcounts.items(),desc="Selecting point with most outliers"):
        # # if g_idx not in terrain_gidxs:
        # if g_idx in terrain_gidxs:
        #     continue
        # # if count < skip_count:
        # #     count += 1
        # #     continue
        # # chosen_gidx = g_idx
        # # clipid_2_counts = cid_2_cnt
        # # break
        
        # ## Remove terrain text embeddings
        # cid_2_cnt_old = dict(cid_2_cnt)
        # for i, (clip_id, count) in enumerate(cid_2_cnt_old.items()):
        #     if clip_id < terra.num_terrain:
        #         del cid_2_cnt[clip_id]
        
        # Build X and w
        X = torch.zeros((len(cid_2_cnt), 512), device=device)
        w = torch.zeros((len(cid_2_cnt),), device=device)

        for i, (clip_id, count) in enumerate(cid_2_cnt.items()):
            # if clip_id < terra.num_terrain:
            #     print(f"Error: clip_id {clip_id} is less than num_terrain {terra.num_terrain}")
            X[i,:] = clip_ids[clip_id,:]
            w[i] = count
        if w.sum() == 0:
            # To handle terrain only embeddings that have all been removed
            continue
        
        num_outliers = count_outliers(X, w, std_dev=3.5)
        # num_outliers = count_outliers(X, w, threshold=0.3)

        gidx_2_outlier_counts[g_idx] = num_outliers.detach().cpu().item()
        gidx_2_total_counts[g_idx] = w.sum().detach().cpu().item()
        
        if num_outliers > max_outliers:
            max_outliers = num_outliers
            chosen_gidx = g_idx
            clipid_2_counts = cid_2_cnt
            print("New max number of outliers",max_outliers)

    # # Compute Global Outlier Ratio
    # num = 0.0
    # den = 0.0
    # min_ratio = 1.0
    # max_ratio = 0.0
    # gidx_2_outlier_ratio = {}
    # for g_idx in gidx_2_outlier_counts:
    #     num += gidx_2_outlier_counts[g_idx]
    #     den += gidx_2_total_counts[g_idx]
    #     ratio = gidx_2_outlier_counts[g_idx] / gidx_2_total_counts[g_idx]
    #     if ratio < min_ratio:
    #         min_ratio = ratio
    #     if ratio > max_ratio:
    #         max_ratio = ratio        
    #     gidx_2_outlier_ratio[g_idx] = ratio

    # # Sort by ratio (descending)
    # sorted_items = sorted(gidx_2_outlier_ratio.items(), key=lambda x: x[1], reverse=True)
    # sorted_idx = [x[0] for x in sorted_items]
    # sorted_ratios = [x[1] for x in sorted_items]
    
    # print("\nGlobal Outlier Ratio:", num / den,"\n")
    # print("Min Outlier Ratio:", min_ratio)
    # print("Max Outlier Ratio:", max_ratio)
    # ratios = np.array(sorted_ratios)
    # print("Mean:", ratios.mean())
    # print("Std:", ratios.std())
    # print("Median:", np.median(ratios))
    # # Top-k concentration
    # sorted_outliers = np.array([gidx_2_outlier_counts[i] for i in sorted_idx])
    # sorted_totals = np.array([gidx_2_total_counts[i] for i in sorted_idx])
    # top_10 = int(0.1 * len(sorted_outliers))
    # top_outliers = sorted_outliers[:top_10].sum()
    # total_outliers = sorted_outliers.sum()
    # print("Top 10% contribution (correct):", top_outliers / total_outliers)
    
    # cum_outliers = np.cumsum(sorted_outliers) / sorted_outliers.sum()
    # half_idx = np.searchsorted(cum_outliers, 0.5)
    # print("Fraction of points that account for 50% of outliers:", half_idx / len(cum_outliers))
    
    # plt.figure(figsize=(6,4))
    # plt.plot(np.linspace(0, 1, len(cum_outliers)), cum_outliers)
    # plt.xlabel("Fraction of points (sorted)")
    # plt.ylabel("Fraction of total outliers")
    # plt.title("Outlier Concentration Curve")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    
    # plt.figure(figsize=(10, 4))
    # plt.bar(range(len(sorted_ratios)), sorted_ratios)
    # plt.xlabel("Points (sorted by outlier ratio)")
    # plt.ylabel("Outlier ratio")
    # plt.title("Sorted Outlier Ratios per Point")
    # plt.tight_layout()
    # plt.show()

    # # Display point cloud colored by outliers
    # vals = np.array(list(gidx_2_outlier_counts.values()))
    
    # plt.figure(figsize=(7, 5))
    # plt.hist(vals, bins=50)
    # plt.xlabel("Number of outliers per point")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of CLIP Outliers per Point")
    # plt.grid(True, alpha=0.3)
    # plt.show()
    
    # # vmin, vmax = vals.min(), vals.max()
    # vmin = np.percentile(vals, 5)
    # vmax = np.percentile(vals, 95)
    
    # pc = terra.pc
    # all_points = pc[:, :3].astype(np.float64)
    # all_colors = np.zeros((pc.shape[0], 3), dtype=np.float64)
    # for g_idx, cid_2_cnt in gidx_2_clipcounts.items():
    #     pts = pc[g_idx,:3]
    #     # normalize score
    #     if g_idx in gidx_2_outlier_counts:
    #         score = gidx_2_outlier_counts[g_idx]
    #     else:
    #         score = vmin
    #     t = (score - vmin) / (vmax - vmin + 1e-8)
    #     t = np.clip(t, 0, 1)
    #     # t = t ** 0.5 # contrast boosting for better visualization
    #     color = jet_colormap(np.array([t]))[0]  # RGB
    #     all_colors[g_idx] = color
    # all_points = np.asarray(all_points, dtype=np.float64).reshape(-1, 3)
    # all_colors = np.asarray(all_colors, dtype=np.float64).reshape(-1, 3)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_points)
    # pcd.colors = o3d.utility.Vector3dVector(all_colors)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # render_opt = vis.get_render_option()
    # render_opt.point_size = 4.0  # smaller points
    # vis.run()
    # vis.destroy_window()

    # print(f"Chosen gidx: {chosen_gidx}")
    # print(f"Number of outliers: {max_outliers}")
        
    
    is_terrain = False
    if chosen_gidx in terrain_gidxs:
        is_terrain = True
        print(f"Chosen global index {chosen_gidx} is in terrain_gidxs")
    else:
        print(f"Chosen global index {chosen_gidx} is NOT in terrain_gidxs")
    
    # Weighted mean (currently: MS-Avg)
    X = torch.zeros((len(clipid_2_counts), 512), device=device)
    w = torch.zeros((len(clipid_2_counts),), device=device)
    mu_w = torch.zeros(512, device=device)
    num_detections = 0
    for i, (clip_id, count) in enumerate(clipid_2_counts.items()):
        X[i,:] = clip_ids[clip_id,:]
        w[i] = count
        mu_w.add_(clip_ids[clip_id,:], alpha=count)
        num_detections += count
    mu_w.div_(num_detections)
    mu_w.div_(mu_w.norm(dim=-1,keepdim=True))
    X.div_(X.norm(dim=-1,keepdim=True))
    
    # diff = X - mu_w.unsqueeze(0) # [num_clip_ids, 512]
    # dist = diff.norm(dim=1) # [num_clip_ids,]
    diff = tensor_cosine_similarity(X, mu_w) # [num_clip_ids,]
    dist = 1 - diff # convert cosine similarity to distance
    
    medoid_w = compute_weighted_medoid(X, w)
    hdbscan_w = compute_hdbscan_embedding(X, w, min_cluster_size=2, selection="weight")
    trimmed_mean_w_02 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.2)
    trimmed_mean_w_04 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.4)
    trimmed_mean_w_06 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.6)
    trimmed_mean_w_09 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.9)
    trimmed_mean_w_095 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.95)
    trimmed_med_w_3_5 = compute_weighted_trimmed_medoid(X, w, z_mod_thresh=3.5)
    # geometric_median_w = compute_weighted_geometric_median(X, w, max_iter=100, tol=1e-5)
    max_w = X[w.argmax(),:]
    
    mean_xy = np.array([mu_w[92].item(), mu_w[133].item()])
    medoid_xy = np.array([medoid_w[92].item(), medoid_w[133].item()])
    trimmed_mean_02_xy = np.array([trimmed_mean_w_02[92].item(), trimmed_mean_w_02[133].item()])
    trimmed_mean_04_xy = np.array([trimmed_mean_w_04[92].item(), trimmed_mean_w_04[133].item()])
    trimmed_mean_06_xy = np.array([trimmed_mean_w_06[92].item(), trimmed_mean_w_06[133].item()])
    trimmed_mean_09_xy = np.array([trimmed_mean_w_09[92].item(), trimmed_mean_w_09[133].item()])
    trimmed_mean_095_xy = np.array([trimmed_mean_w_095[92].item(), trimmed_mean_w_095[133].item()])
    trimmed_med_3_5_xy = np.array([trimmed_med_w_3_5[92].item(), trimmed_med_w_3_5[133].item()])
    # geometric_median_xy = np.array([geometric_median_w[92].item(), geometric_median_w[133].item()])
    max_xy = np.array([max_w[92].item(), max_w[133].item()])
    
    w_np = w.detach().cpu().numpy()
    w_min, w_max = w_np.min(), w_np.max()
    if w_max > w_min:
        w_norm = (w_np - w_min) / (w_max - w_min)
    else:
        w_norm = np.ones_like(w_np)
    # Scale to reasonable marker sizes (area in points^2)
    min_size = 20
    max_size = 400
    sizes = min_size + w_norm * (max_size - min_size)
    xys = np.stack([X[:,92].detach().cpu().numpy(), X[:,133].detach().cpu().numpy()], axis=1)
    
    # Display distribution of points using dimensions 133 and 92
    plt.figure()
    plt.scatter(xys[:,0], xys[:,1], alpha=0.5, label="Clip Embeddings", s=sizes)
    plt.scatter(mean_xy[0], mean_xy[1], color='red', label='Weighted Mean', marker='X', s=200)
    plt.scatter(medoid_xy[0], medoid_xy[1], color='green', label='Weighted Medoid', marker='^', s=200)
    # plt.scatter(geometric_median_xy[0], geometric_median_xy[1], color='purple', label='Weighted Geometric Median', marker='>', s=200)
    plt.scatter(max_xy[0], max_xy[1], color='brown', label='Max Embedding', marker='P', s=200)
    plt.scatter(trimmed_mean_02_xy[0], trimmed_mean_02_xy[1], color='orange', label='Weighted Trimmed Mean t=0.2', marker='*', s=200)
    plt.scatter(trimmed_mean_04_xy[0], trimmed_mean_04_xy[1], color='magenta', label='Weighted Trimmed Mean t=0.4', marker='*', s=200)
    plt.scatter(trimmed_mean_06_xy[0], trimmed_mean_06_xy[1], color='green', label='Weighted Trimmed Mean t=0.6', marker='*', s=200)
    plt.scatter(trimmed_mean_09_xy[0], trimmed_mean_09_xy[1], color='blue', label='Weighted Trimmed Mean t=0.9', marker='*', s=200)
    plt.scatter(trimmed_mean_095_xy[0], trimmed_mean_095_xy[1], color='black', label='Weighted Trimmed Mean t=0.95', marker='*', s=200)
    plt.scatter(trimmed_med_3_5_xy[0], trimmed_med_3_5_xy[1], color='cyan', label='Weighted Trimmed Medoid z=3.5', marker='D', s=200)
    plt.xlabel("Embedding Dimension 92")
    plt.ylabel("Embedding Dimension 133")
    if is_terrain:
        plt.title(f"Clip Embeddings and Averages for Terrain Global Index {chosen_gidx}")
    else:
        plt.title(f"Clip Embeddings and Averages for Nonterrain Global Index {chosen_gidx}")
    plt.legend()
    plt.show()
      
    # Display distribution of distances from mean (do outliers exist?)
    plt.figure()
    plt.hist(dist.cpu().detach().numpy(), bins=50)
    plt.xlabel("Distance from Weighted Mean", fontsize=17)
    plt.ylabel("Frequency", fontsize=17)
    plt.title(f"Distance Distribution from Mean", fontsize=18)# for Global Index {chosen_gidx}")
    plt.tick_params(axis='both', labelsize=15)
    plt.show()
    
    # # Display distribution of distances from medoid, trimmed mean, geometric median
    # diff_medoid = X - medoid_w.unsqueeze(0)
    # dist_medoid = diff_medoid.norm(dim=1)
    # plt.hist(dist_medoid.cpu().detach().numpy(), bins=50)
    # plt.xlabel("Distance from Weighted Medoid")
    # plt.ylabel("Frequency")
    # plt.title(f"Distance Distribution from Medoid for Global Index {chosen_gidx}")
    # plt.show()
    
    # diff_trimmed_mean = X - trimmed_mean_w.unsqueeze(0)
    # dist_trimmed_mean = diff_trimmed_mean.norm(dim=1)
    # plt.hist(dist_trimmed_mean.cpu().detach().numpy(), bins=50)
    # plt.xlabel("Distance from Weighted Trimmed Mean")   
    # plt.ylabel("Frequency")
    # plt.title(f"Distance Distribution from Trimmed Mean for Global Index {chosen_gidx}")
    # plt.show()
    
    # diff_geometric_median = X - geometric_median_w.unsqueeze(0)
    # dist_geometric_median = diff_geometric_median.norm(dim=1)
    # plt.hist(dist_geometric_median.cpu().detach().numpy(), bins=50)
    # plt.xlabel("Distance from Weighted Geometric Median")
    # plt.ylabel("Frequency")
    # plt.title(f"Distance Distribution from Geometric Median for Global Index {chosen_gidx}")
    # plt.show()