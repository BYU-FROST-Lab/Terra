from argparse import ArgumentParser
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

import open3d as o3d

from utils import tensor_cosine_similarity

# def geometric_median(X, w=None, eps=1e-5, max_iter=100):
#     """
#     X: (N, D)
#     w: (N,) or None
#     """

#     # Normalize embeddings (same as your pipeline)
#     X = X / X.norm(dim=1, keepdim=True)

#     N, D = X.shape

#     # ---- FAST PATH: no weights or uniform weights ----
#     if w is None or torch.all(w == 1):
#         y = X.mean(dim=0)

#         for _ in range(max_iter):
#             diff = X - y
#             dist = torch.norm(diff, dim=1).clamp(min=eps)

#             weights = 1.0 / dist
#             y_new = (weights[:, None] * X).sum(dim=0) / weights.sum()

#             if torch.norm(y - y_new) < eps:
#                 break
#             y = y_new

#     # ---- Weighted case ----
#     else:
#         w = w / w.sum()
#         y = (X * w.unsqueeze(1)).sum(dim=0)

#         for _ in range(max_iter):
#             diff = X - y
#             dist = torch.norm(diff, dim=1).clamp(min=eps)

#             weights = w / dist
#             y_new = (weights[:, None] * X).sum(dim=0) / weights.sum()

#             if torch.norm(y - y_new) < eps:
#                 break
#             y = y_new

#     return y / y.norm()

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

def compute_mean_distance(X, w=None):
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

    return dist.mean().item()

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

def jet_colormap(t):
    # t in [0,1]
    r = np.clip(1.5 - np.abs(4*t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*t - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)

def display_gidx(g_idx, pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    colors = np.ones_like(pc[:,:3]) * 0.5
    colors[g_idx] = np.array([1.0, 0.0, 0.0]) # Highlight the point in red
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Add sphere at gidx point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    sphere.translate(pc[g_idx, :3])
    sphere.paint_uniform_color([1.0, 0.0, 0.0]) # Set base color to gray
    o3d.visualization.draw_geometries([pcd] + [sphere])

def plot_dist(title, g_idx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar):
    fig, ax = plt.subplots(figsize=(8, 6))
    curr_ratio = gidx_2_outlier_ratio[g_idx]
    
    xs = []
    ys = []
    for (med_dist, lidar_dist) in zip(gidx_2_dists_from_med[g_idx], gidx_2_dists_from_lidar[g_idx]):
        xs.append(lidar_dist)
        ys.append(med_dist)
    colors = np.arange(len(xs))
    scatter = ax.scatter(
        xs,
        ys,
        marker='o',
        c=colors,
        cmap='jet',
        s=50,
        alpha=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Observation Index", fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Distance from LiDAR", fontsize=15)
    ax.set_ylabel("Distance from Median", fontsize=15)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.show()

def plot_fastsam_masks(g_idx, mask_idxs, fsam_mask_names, output_dir):
    mask_names = [fsam_mask_names[idx] for idx in mask_idxs]
    imgs_per_fig = 9
    num_plots = 0
    for start_idx in range(0, len(mask_names), imgs_per_fig):
        curr_mask_names = mask_names[start_idx:start_idx + imgs_per_fig]
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        for ax_idx, ax in enumerate(axes):
            # Hide unused axes
            if ax_idx >= len(curr_mask_names):
                ax.axis("off")
                continue
            mask_path = output_dir.parent.parent / curr_mask_names[ax_idx]

            # Load image
            mask_idx = int(start_idx) + int(ax_idx)
            # print(f"Loading mask {mask_idx}: {mask_path}")
            img = plt.imread(mask_path)
            ax.imshow(img)
            ax.set_title(
                f"{mask_idx}:{Path(mask_path).stem}",
                fontsize=8
            )
            ax.axis("off")
        fig.suptitle(
            f"FastSAM Masks for Global Index {g_idx}",
            fontsize=18
        )
        fig.tight_layout()
        num_plots += 1
        if num_plots > 40:
            break
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to load the saved msmap output files.")
    parser.add_argument('--clip_segs', type=str, required=True, help="Tensor of segment CLIP embeddings.")
    parser.add_argument('--gidx2clipcounts', type=str, required=True, help="Dictionary mapping global indices to clip counts")
    parser.add_argument('--global_pc', type=str, required=True, help="Global point cloud file")
    parser.add_argument('--gidx2clipdists', type=str, help="Dictionary mapping global indices to clip_ids with LiDAR distances")
    parser.add_argument('--gidx2clipmaskidx', type=str, help="Dictionary mapping global indices to clip_ids with mask indices")
    parser.add_argument('--fsam_mask_names', type=str, help="List of mask names corresponding to clipmaskidx")
    # parser.add_argument('--clip_imgs', type=str, help="Tensor of image CLIP embeddings")
    # parser.add_argument('--gidx2imgs', type=str, help="Dictionary mapping global indices to image indices within distance")
    # parser.add_argument('--img_names', type=str, help="List of image names corresponding to clip_imgs")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    # Load CLIP Segment data
    output_dir = Path(args.output_dir)
    with open(output_dir / args.clip_segs, "rb") as f:
        clip_ids = torch.load(f)
    with open(output_dir / args.gidx2clipcounts, "rb") as f:
        gidx_2_clipcounts = pkl.load(f)
    with open(output_dir / args.gidx2clipdists, "rb") as f:
        gidx_2_clipdists = pkl.load(f)
    with open(output_dir / args.gidx2clipmaskidx, "rb") as f:
        gidx_2_clipmaskidx = pkl.load(f)
    with open(output_dir / args.fsam_mask_names, "rb") as f:
        fsam_mask_names = pkl.load(f)
    pc = np.load(args.global_pc) # (num_pts,4)
    
    max_outliers = -1
    chosen_gidx = None
    clipid_2_counts = None
    
    gidx_2_outlier_counts = {}
    gidx_2_total_counts = {}
    # gidx_2_avg_dist = {}
    gidx_2_dists_from_med = {}
    gidx_2_dists_from_lidar = {}
    max_obsv = 0
    # count_cutoff = 1000
    # count_idx = 0
    gidx_list = []
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
        if len(dists) > max_obsv:
            max_obsv = len(dists)
            
        # gidx_2_avg_dist[g_idx] = compute_mean_distance(X, w=None)
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
        
        # count_idx += 1
        # if count_idx >= count_cutoff:
        #     break

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

    # Sort by ratio (ascending)
    sorted_ratio_items = sorted(gidx_2_outlier_ratio.items(), key=lambda x: x[1], reverse=False)
    sorted_idx = [x[0] for x in sorted_ratio_items]
    sorted_ratios = [x[1] for x in sorted_ratio_items]
    sorted_obs = [gidx_2_total_counts[i] for i in sorted_idx]
    
    sorted_outlier_counts_items = sorted(gidx_2_outlier_counts.items(), key=lambda x: x[1], reverse=False)
    sorted_gidx_by_outliers = [x[0] for x in sorted_outlier_counts_items]
    sorted_outlier_counts = [x[1] for x in sorted_outlier_counts_items]

    # x = np.linspace(0, 1, len(sorted_ratios))
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(7,6),
        sharex=True
    )

    # Top plot: outlier ratio
    ax1.plot(range(len(sorted_ratios)), sorted_ratios)
    ax1.set_ylabel("Outlier Ratio",fontsize=15)
    ax1.set_title("Point Cloud Outlier Presence",fontsize=18)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.grid(True)

    # Bottom plot: number of observations
    ax2.plot(range(len(sorted_ratios)), sorted_obs)
    ax2.set_xlabel("Point Index",fontsize=15)
    ax2.set_ylabel("Observation Count",fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)
    ax2.tick_params(axis='x', labelrotation=45)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
    # ####################################################
    # ## FULL POINT CLOUD SPATIAL v. SEMANTIC DISTANCES ##
    # ####################################################
    # gidx_sampled = np.random.choice(gidx_list, size=min(10, len(gidx_list)), replace=False)
    # max_obsv = max([len(gidx_2_clipcounts[g_idx]) for g_idx in gidx_sampled])
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # for g_idx in gidx_sampled:
    #     curr_ratio = gidx_2_outlier_ratio[g_idx]
    #     # if curr_ratio < 0.1:
    #     #     continue
    #     t = np.clip(curr_ratio / 0.5, 0, 1)
    #     color = jet_colormap(np.array([t]))[0]  # RGB
        
    #     # # Color by number of observations
    #     # t = np.clip(len(gidx_2_dists_from_med[g_idx]) / max_obsv, 0, 1)
    #     # color = jet_colormap(np.array([t]))[0]
        
    #     xs = []
    #     ys = []
    #     for (med_dist, lidar_dist) in zip(gidx_2_dists_from_med[g_idx], gidx_2_dists_from_lidar[g_idx]):
    #         xs.append(lidar_dist)
    #         ys.append(med_dist)

    #     ax.scatter(
    #         xs,
    #         ys,
    #         marker='o',
    #         # color='k',
    #         color=color,
    #         s=30,
    #         alpha=0.5
    #     )
    
    # norm = mcolors.Normalize(vmin=0.0, vmax=max_obsv)#0.5)
    # cmap = cm.jet
    # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax)
    # # cbar.set_label("Outlier Ratio", fontsize=15)
    # cbar.set_label("Number of Observations", fontsize=15)
    # cbar.ax.tick_params(labelsize=15)

    # ax.set_title("Semantic vs Geometric Distances", fontsize=18)
    # ax.set_xlabel("Distance from LiDAR", fontsize=15)
    # ax.set_ylabel("Distance from Median", fontsize=15)

    # ax.grid(True)
    # ax.tick_params(axis='both', labelsize=15)

    # plt.tight_layout()
    # plt.show()
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # for g_idx in gidx_sampled:
    #     curr_ratio = gidx_2_outlier_ratio[g_idx]
    #     t = np.clip(curr_ratio / 0.5, 0, 1)
        
    #     xs = []
    #     ys = []
    #     for (med_dist, lidar_dist) in zip(gidx_2_dists_from_med[g_idx], gidx_2_dists_from_lidar[g_idx]):
    #         xs.append(lidar_dist)
    #         ys.append(med_dist)

    #     ax.scatter(
    #         xs,
    #         ys,
    #         marker='o',
    #         color='k',
    #         s=50,
    #         alpha=0.2
    #     )

    # ax.set_title("Semantic vs Geometric Distances", fontsize=18)
    # ax.set_xlabel("Distance from LiDAR", fontsize=15)
    # ax.set_ylabel("Distance from Median", fontsize=15)

    # ax.grid(True)
    # ax.tick_params(axis='both', labelsize=15)

    # plt.tight_layout()
    # plt.show()
    
    ##################################
    ## FastSAM Masks (with >2 Obsv) ##
    ##################################
    spread_geomtrically = {}
    spread_semantically = {}
    for g_idx in gidx_list:
        if len(gidx_2_clipcounts[g_idx]) <= 2:
            continue
        spread_geomtrically[g_idx] = np.max(gidx_2_dists_from_lidar[g_idx]) - np.min(gidx_2_dists_from_lidar[g_idx])
        spread_semantically[g_idx] = np.max(gidx_2_dists_from_med[g_idx]) - np.min(gidx_2_dists_from_med[g_idx])
    gidxs = np.array(list(spread_geomtrically.keys()))
    geom_spreads = np.array([spread_geomtrically[g] for g in gidxs])
    sem_spreads  = np.array([spread_semantically[g] for g in gidxs])
    geom_norm = (geom_spreads - geom_spreads.min()) / (
        geom_spreads.max() - geom_spreads.min()
    )
    sem_norm = (sem_spreads - sem_spreads.min()) / (
        sem_spreads.max() - sem_spreads.min()
    )
    
    # Select a point with the largest spread geometrically and semantically
    score_lgls = geom_norm + sem_norm
    lgls_gidx = gidxs[np.argmax(score_lgls)]  
    display_gidx(lgls_gidx, pc)
    # mask_idxs = list(set().union(*gidx_2_clipmaskidx[lgls_gidx].values())) # changes order. BAD!
    mask_idxs = []
    for mask_idx_set in gidx_2_clipmaskidx[lgls_gidx].values():
        mask_idxs.extend(list(mask_idx_set))
    plot_fastsam_masks(lgls_gidx, mask_idxs, fsam_mask_names, output_dir)
    plot_dist("Large Geom + Large Sem", lgls_gidx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar)
    
    # Select a point with the largest spread geometrically but smallest semantically
    score_lgss = geom_norm - sem_norm
    lgss_gidx = gidxs[np.argmax(score_lgss)]
    display_gidx(lgss_gidx, pc)
    mask_idxs = []
    for mask_idx_set in gidx_2_clipmaskidx[lgss_gidx].values():
        mask_idxs.extend(list(mask_idx_set))
    plot_fastsam_masks(lgss_gidx, mask_idxs, fsam_mask_names, output_dir)
    plot_dist("Large Geom + Small Sem", lgss_gidx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar)
    
    # Select a point with the largest spread semantically but smallest geometrically
    score_lssl = sem_norm - geom_norm
    sgls_gidx = gidxs[np.argmax(score_lssl)]
    display_gidx(sgls_gidx, pc)
    mask_idxs = []
    for mask_idx_set in gidx_2_clipmaskidx[sgls_gidx].values():
        mask_idxs.extend(list(mask_idx_set))
    plot_fastsam_masks(sgls_gidx, mask_idxs, fsam_mask_names, output_dir)
    plot_dist("Small Geom + Large Sem", sgls_gidx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar)
    
    # Select a point with the smallest spread semantically and geometrically
    score_ss = sem_norm + geom_norm
    sgss_gidx = gidxs[np.argmin(score_ss)]
    display_gidx(sgss_gidx, pc)
    mask_idxs = []
    for mask_idx_set in gidx_2_clipmaskidx[sgss_gidx].values():
        mask_idxs.extend(list(mask_idx_set))
    plot_fastsam_masks(sgss_gidx, mask_idxs, fsam_mask_names, output_dir)
    plot_dist("Small Geom + Small Sem", sgss_gidx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar)

    ## Plot 4 selected points
    fig, ax = plt.subplots(figsize=(8, 6))

    gidx_infos = [
        (lgls_gidx, "Large Geom + Large Sem", "red"),
        (lgss_gidx, "Large Geom + Small Sem", "blue"),
        (sgls_gidx, "Small Geom + Large Sem", "green"),
        (sgss_gidx, "Small Geom + Small Sem", "purple"),
    ]

    for g_idx, label, color in gidx_infos:

        xs = []
        ys = []

        for (med_dist, lidar_dist) in zip(
            gidx_2_dists_from_med[g_idx],
            gidx_2_dists_from_lidar[g_idx]
        ):
            xs.append(lidar_dist)
            ys.append(med_dist)

        ax.scatter(
            xs,
            ys,
            marker='o',
            color=color,
            s=50,
            alpha=0.4,
            label=label
        )

    ax.set_title("Semantic vs Geometric Distances", fontsize=18)
    ax.set_xlabel("Distance from LiDAR", fontsize=15)
    ax.set_ylabel("Distance from Median", fontsize=15)

    ax.grid(True)
    ax.tick_params(axis='both', labelsize=15)

    # Prevent duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        fontsize=12
    )

    plt.tight_layout()
    plt.show()
    
    ## Displaying by outliers
    valid_gidxs = (
        set(gidx_2_clipmaskidx.keys())
        & set(gidx_2_outlier_ratio.keys())
        & set(gidx_2_dists_from_med.keys())
        & set(gidx_2_dists_from_lidar.keys())
        & set(gidx_2_outlier_counts.keys())
    )
    filtered_ratio_pairs = [
        (g_idx, ratio)
        for g_idx, ratio in zip(sorted_idx, sorted_ratios)
        if g_idx in valid_gidxs
    ]
    filtered_count_pairs = [
        (g_idx, count)
        for g_idx, count in zip(sorted_gidx_by_outliers, sorted_outlier_counts)
        if g_idx in valid_gidxs
    ]
    topk = 5
    
    min_or_pairs = filtered_ratio_pairs[:topk]
    max_or_pairs = filtered_ratio_pairs[-topk:]

    min_or_gidxs = [x[0] for x in min_or_pairs]
    min_outlier_ratios = [x[1] for x in min_or_pairs]

    max_or_gidxs = [x[0] for x in max_or_pairs]
    max_outlier_ratios = [x[1] for x in max_or_pairs]

    # ---------------------------------------------------------
    # Count selections (filtered)
    # ---------------------------------------------------------
    min_oc_pairs = filtered_count_pairs[:topk]
    max_oc_pairs = filtered_count_pairs[-topk:]

    min_oc_gidxs = [x[0] for x in min_oc_pairs]
    min_outlier_counts = [x[1] for x in min_oc_pairs]

    max_oc_gidxs = [x[0] for x in max_oc_pairs]
    max_outlier_counts = [x[1] for x in max_oc_pairs]
    
    # Display top-k by outlier ratio
    for i, g_idx in enumerate(max_or_gidxs):
        print(f"Top-{topk - i} High Outlier Ratio Global Index:", g_idx)
        display_gidx(g_idx, pc)
        mask_idxs = []
        for mask_idx_set in gidx_2_clipmaskidx[g_idx].values():
            mask_idxs.extend(list(mask_idx_set))
        plot_fastsam_masks(g_idx, mask_idxs, fsam_mask_names, output_dir)
        plot_dist(f"High Outlier Ratio {gidx_2_outlier_ratio[g_idx]:.2f}", g_idx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar)
    
    for i, g_idx in enumerate(min_or_gidxs):
        print(f"Top-{i+1} Low Outlier Ratio Global Index:", g_idx)
        display_gidx(g_idx, pc)
        mask_idxs = []
        for mask_idx_set in gidx_2_clipmaskidx[g_idx].values():
            mask_idxs.extend(list(mask_idx_set))
        plot_fastsam_masks(g_idx, mask_idxs, fsam_mask_names, output_dir)
        plot_dist(f"Low Outlier Ratio {gidx_2_outlier_ratio[g_idx]:.2f}", g_idx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar)
    
    # Display top-k by outlier count
    for i, g_idx in enumerate(max_oc_gidxs):
        print(f"Top-{topk - i} High Outlier Count Global Index:", g_idx)
        display_gidx(g_idx, pc)
        mask_idxs = []
        for mask_idx_set in gidx_2_clipmaskidx[g_idx].values():
            mask_idxs.extend(list(mask_idx_set))
        plot_fastsam_masks(g_idx, mask_idxs, fsam_mask_names, output_dir)
        plot_dist(f"High Outlier Count {gidx_2_outlier_counts[g_idx]}", g_idx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar)
    
    for i, g_idx in enumerate(min_oc_gidxs):
        print(f"Top-{i+1} Low Outlier Count Global Index:", g_idx)
        display_gidx(g_idx, pc)
        mask_idxs = []
        for mask_idx_set in gidx_2_clipmaskidx[g_idx].values():
            mask_idxs.extend(list(mask_idx_set))
        plot_fastsam_masks(g_idx, mask_idxs, fsam_mask_names, output_dir)
        plot_dist(f"Low Outlier Count {gidx_2_outlier_counts[g_idx]}", g_idx, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar)

    # ## Spatial & Semantic Distance graph
    # # Select 3 random inlier points and 3 random outlier points for visualization
    # np.random.seed(42)
    # num_samples = 3
    # high_ratio_points = np.array([
    #     g_idx for g_idx, ratio in gidx_2_outlier_ratio.items()
    #     if ratio > 0.3 and len(gidx_2_clipcounts[g_idx]) > 10
    # ])
    # low_ratio_points = np.array([
    #     g_idx for g_idx, ratio in gidx_2_outlier_ratio.items()
    #     if ratio < 0.1 and len(gidx_2_clipcounts[g_idx]) > 10
    # ])
    # # Randomly select up to 3 from each group
    # selected_high = np.random.choice(
    #     high_ratio_points,
    #     size=min(num_samples, len(high_ratio_points)),
    #     replace=False
    # )
    # selected_low = np.random.choice(
    #     low_ratio_points,
    #     size=min(num_samples, len(low_ratio_points)),
    #     replace=False
    # )
    
    # labels = []
    
    # print("High ratio points (>0.3):", selected_high)
    # dists_from_med_high = []
    # dists_from_lidar_high = []
    # for g_idx in selected_high:
    #     print(f"Point {g_idx} - Outlier Ratio: {gidx_2_outlier_ratio[g_idx]:.2f}")
    #     print(f"  Number of outliers: {gidx_2_outlier_counts[g_idx]}, Number of observations: {gidx_2_total_counts[g_idx]}\n")
    #     labels.append(f"Ratio {gidx_2_outlier_counts[g_idx]}/{gidx_2_total_counts[g_idx]}")
    #     X = torch.zeros((0, 512), device=device)
    #     dists_from_lidar = []
    #     for cid, dist in gidx_2_clipdists[g_idx].items():
    #         if cid < 7: # Assuming terrain clip IDs are 0-6, adjust if needed
    #             continue
    #         # print(f"  Clip ID {cid} - LiDAR Distance: {dist:.2f}")
    #         X = torch.cat((X, clip_ids[cid,:].unsqueeze(0)), dim=0)
    #         dists_from_lidar.append(dist)
        
    #     # Normalize embeddings
    #     X = X / X.norm(dim=1, keepdim=True)

    #     mu = X.mean(dim=0)
    #     mu = mu / mu.norm()

    #     sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
    #     dists = 1 - sim

    #     med = dists.median()        
    #     dists_from_med = torch.abs(dists - med)
    #     # dists_from_med = dists
        
    #     dists_from_med_high.append(dists_from_med.cpu().detach().numpy())
    #     dists_from_lidar_high.append(dists_from_lidar)
    
    # print("Low ratio points (<0.1):", selected_low)
    # dists_from_med_low = []
    # dists_from_lidar_low = []
    # for g_idx in selected_low:
    #     print(f"Point {g_idx} - Outlier Ratio: {gidx_2_outlier_ratio[g_idx]:.2f}")
    #     print(f"  Number of outliers: {gidx_2_outlier_counts[g_idx]}, Number of observations: {gidx_2_total_counts[g_idx]}\n")
    #     labels.append(f"Ratio {gidx_2_outlier_counts[g_idx]}/{gidx_2_total_counts[g_idx]}")
    #     X = torch.zeros((0, 512), device=device)
    #     dists_from_lidar = []
    #     for cid, dist in gidx_2_clipdists[g_idx].items():
    #         if cid < 7: # Assuming terrain clip IDs are 0-6, adjust if needed
    #             continue
    #         # print(f"  Clip ID {cid} - LiDAR Distance: {dist:.2f}")
    #         X = torch.cat((X, clip_ids[cid,:].unsqueeze(0)), dim=0)
    #         dists_from_lidar.append(dist)
        
    #     # Normalize embeddings
    #     X = X / X.norm(dim=1, keepdim=True)

    #     mu = X.mean(dim=0)
    #     mu = mu / mu.norm()

    #     sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
    #     dists = 1 - sim

    #     med = dists.median()        
    #     dists_from_med = torch.abs(dists - med)
    #     # dists_from_med = dists
        
    #     dists_from_med_low.append(dists_from_med.cpu().detach().numpy())
    #     dists_from_lidar_low.append(dists_from_lidar)
    
    
    # fig, ax = plt.subplots(figsize=(8, num_samples*2))
    # marker_styles = ['o', 's', '^', 'D', 'P', 'X']
    # colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    # # labels = [
    # #     'High Ratio 1', 'High Ratio 2', 'High Ratio 3',
    # #     'Low Ratio 1', 'Low Ratio 2', 'Low Ratio 3'
    # # ]
    # for i in range(num_samples*2):
    #     if i < len(dists_from_med_high):
    #         x = dists_from_lidar_high[i]
    #         y = dists_from_med_high[i]
    #     else:
    #         x = dists_from_lidar_low[i - num_samples]
    #         y = dists_from_med_low[i - num_samples]
        
    #     ax.scatter(
    #         x,
    #         y,
    #         marker=marker_styles[i],
    #         color=colors[i],
    #         s=80,
    #         alpha=0.8,
    #         label=labels[i]
    #     )

    # ax.set_title("Semantic vs Geometric Distances", fontsize=18)
    # ax.set_xlabel("Distance from LiDAR", fontsize=15)
    # ax.set_ylabel("Distance from Median", fontsize=15)
    # # ax.set_ylabel("Distance from Mean", fontsize=15)

    # ax.legend(fontsize=15)
    # ax.grid(True)
    # ax.tick_params(axis='both', labelsize=15)

    # plt.tight_layout()
    # plt.show()


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
    # plt.xlabel("Point Index", fontsize=15)
    # plt.ylabel("Outlier ratio", fontsize=15)
    # plt.title("Sorted Outlier Ratios per Point", fontsize=18)
    # plt.tick_params(axis='both', labelsize=15)
    # plt.tick_params(axis='x', labelrotation=45)
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
    
    # pc = np.load(args.global_pc) # (num_pts,4)
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
    
    # ## Use average distance from mean as score instead of outlier count
    # # Display point cloud colored by outliers
    # vals = np.array(list(gidx_2_avg_dist.values()))
    
    # plt.figure(figsize=(7, 5))
    # plt.hist(vals, bins=50)
    # plt.xlabel("Average distance from mean embedding")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of CLIP Average Distances per Point")
    # plt.grid(True, alpha=0.3)
    # plt.show()

    # vmin = np.percentile(vals, 5)
    # vmax = np.percentile(vals, 95)
    
    # pc = np.load(args.global_pc) # (num_pts,4)
    # all_points = pc[:, :3].astype(np.float64)
    # all_colors = np.zeros((pc.shape[0], 3), dtype=np.float64)
    # for g_idx, cid_2_cnt in gidx_2_clipcounts.items():
    #     pts = pc[g_idx,:3]
    #     # normalize score
    #     if g_idx in gidx_2_avg_dist:
    #         score = gidx_2_avg_dist[g_idx]
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