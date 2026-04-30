from argparse import ArgumentParser
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

import open3d as o3d

from utils import tensor_cosine_similarity


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
    parser.add_argument('--clip_segs', type=str, required=True, help="Tensor of segment CLIP embeddings.")
    parser.add_argument('--gidx2clipcounts', type=str, required=True, help="Dictionary mapping global indices to clip counts")
    # parser.add_argument('--clip_imgs', type=str, help="Tensor of image CLIP embeddings")
    # parser.add_argument('--gidx2imgs', type=str, help="Dictionary mapping global indices to image indices within distance")
    # parser.add_argument('--img_names', type=str, help="List of image names corresponding to clip_imgs")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    # Load CLIP Segment data
    with open(args.clip_segs, "rb") as f:
        clip_ids = torch.load(f)
    with open(args.gidx2clipcounts, "rb") as f:
        gidx_2_clipcounts = pkl.load(f)
    
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

    # Sort by ratio (descending)
    sorted_items = sorted(gidx_2_outlier_ratio.items(), key=lambda x: x[1], reverse=True)
    sorted_idx = [x[0] for x in sorted_items]
    sorted_ratios = [x[1] for x in sorted_items]
    
    print("\nGlobal Outlier Ratio:", num / den,"\n")
    print("Min Outlier Ratio:", min_ratio)
    print("Max Outlier Ratio:", max_ratio)
    ratios = np.array(sorted_ratios)
    print("Mean:", ratios.mean())
    print("Std:", ratios.std())
    print("Median:", np.median(ratios))
    # Top-k concentration
    sorted_outliers = np.array([gidx_2_outlier_counts[i] for i in sorted_idx])
    sorted_totals = np.array([gidx_2_total_counts[i] for i in sorted_idx])
    top_10 = int(0.1 * len(sorted_outliers))
    top_outliers = sorted_outliers[:top_10].sum()
    total_outliers = sorted_outliers.sum()
    print("Top 10% contribution (correct):", top_outliers / total_outliers)
    
    cum_outliers = np.cumsum(sorted_outliers) / sorted_outliers.sum()
    half_idx = np.searchsorted(cum_outliers, 0.5)
    print("Fraction of points that account for 50% of outliers:", half_idx / len(cum_outliers))
    
    plt.figure(figsize=(6,4))
    plt.plot(np.linspace(0, 1, len(cum_outliers)), cum_outliers)
    plt.xlabel("Fraction of points (sorted)")
    plt.ylabel("Fraction of total outliers")
    plt.title("Outlier Concentration Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(sorted_ratios)), sorted_ratios)
    plt.xlabel("Points (sorted by outlier ratio)")
    plt.ylabel("Outlier ratio")
    plt.title("Sorted Outlier Ratios per Point")
    plt.tight_layout()
    plt.show()

    # Display point cloud colored by outliers
    vals = np.array(list(gidx_2_outlier_counts.values()))
    
    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=50)
    plt.xlabel("Number of outliers per point")
    plt.ylabel("Frequency")
    plt.title("Distribution of CLIP Outliers per Point")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # vmin, vmax = vals.min(), vals.max()
    vmin = np.percentile(vals, 5)
    vmax = np.percentile(vals, 95)
    
    pc = terra.pc
    all_points = pc[:, :3].astype(np.float64)
    all_colors = np.zeros((pc.shape[0], 3), dtype=np.float64)
    for g_idx, cid_2_cnt in gidx_2_clipcounts.items():
        pts = pc[g_idx,:3]
        # normalize score
        if g_idx in gidx_2_outlier_counts:
            score = gidx_2_outlier_counts[g_idx]
        else:
            score = vmin
        t = (score - vmin) / (vmax - vmin + 1e-8)
        t = np.clip(t, 0, 1)
        # t = t ** 0.5 # contrast boosting for better visualization
        color = jet_colormap(np.array([t]))[0]  # RGB
        all_colors[g_idx] = color
    all_points = np.asarray(all_points, dtype=np.float64).reshape(-1, 3)
    all_colors = np.asarray(all_colors, dtype=np.float64).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 4.0  # smaller points
    vis.run()
    vis.destroy_window()

    print(f"Chosen gidx: {chosen_gidx}")
    print(f"Number of outliers: {max_outliers}")
    