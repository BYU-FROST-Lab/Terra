from argparse import ArgumentParser
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math
import itertools
import yaml

import hdbscan
import open3d as o3d


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

def run_hdbscan_clustering(chosen_gidx, gidx_2_clipcounts, clip_ids, gidx_2_outlier_ratio, gidx_2_dists_from_med, gidx_2_dists_from_lidar):
    num_embs = len(gidx_2_clipcounts[chosen_gidx])
    X = torch.zeros((num_embs, 512), device=device)
    for i, (clip_id, _) in enumerate(gidx_2_clipcounts[chosen_gidx].items()):
        # if clip_id < terra.num_terrain:
        #     print(f"Error: clip_id {clip_id} is less than num_terrain {terra.num_terrain}")
        X[i,:] = clip_ids[clip_id,:]
    # normalize
    X = X / torch.linalg.norm(X, axis=1, keepdims=True)

    hdb = hdbscan.HDBSCAN(
        metric='euclidean',
        min_cluster_size=2,
        min_samples=2, #2
    )

    labels = hdb.fit_predict(X.detach().cpu().numpy())
    
    return labels

def plot_hdbscan_dist(
    title,
    g_idx,
    labels,
    gidx_2_outlier_ratio,
    gidx_2_dists_from_med,
    gidx_2_dists_from_lidar
):
    fig, ax = plt.subplots(figsize=(8, 6))

    xs = []
    ys = []

    for med_dist, lidar_dist in zip(
        gidx_2_dists_from_med[g_idx],
        gidx_2_dists_from_lidar[g_idx]
    ):
        xs.append(lidar_dist)
        ys.append(med_dist)

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)

    non_noise_labels = [l for l in unique_labels if l != -1]

    if len(non_noise_labels) > 0:
        norm = mcolors.Normalize(
            vmin=min(non_noise_labels),
            vmax=max(non_noise_labels)
        )
        cmap = cm.get_cmap("jet")
    else:
        norm = None
        cmap = None

    for label in unique_labels:

        mask = labels == label

        if label == -1:
            color = "black"
            curr_label = "Noise"
        else:
            color = cmap(norm(label))
            curr_label = f"Cluster {label}"

        ax.scatter(
            xs[mask],
            ys[mask],
            c=[color],
            label=curr_label,
            marker='o',
            s=50,
            alpha=0.7
        )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Distance from LiDAR", fontsize=15)
    ax.set_ylabel("Distance from Median", fontsize=15)

    ax.grid(True)
    ax.tick_params(axis='both', labelsize=15)

    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()   
  
def plot_fastsam_masks_by_cluster(
    g_idx,
    labels,
    gidx_2_clipcounts,
    gidx_2_clipmaskidx,
    fsam_mask_names,
    output_dir,
    imgs_per_fig=9,
    max_figs_per_cluster=20
):
    """
    Display FastSAM masks grouped by HDBSCAN cluster labels.

    Assumes labels are in the SAME ORDER as:
        gidx_2_clipcounts[g_idx].items()

    Noise points are cluster label == -1.
    """

    labels = np.asarray(labels)

    # Maintain exact ordering used during HDBSCAN embedding creation
    ordered_clip_ids = [
        clip_id
        for clip_id, _ in gidx_2_clipcounts[g_idx].items()
    ]

    unique_labels = np.unique(labels)

    for cluster_label in unique_labels:

        cluster_mask_idxs = []

        # Collect all mask indices belonging to this cluster
        for clip_idx, clip_id in enumerate(ordered_clip_ids):

            if labels[clip_idx] != cluster_label:
                continue

            if clip_id not in gidx_2_clipmaskidx[g_idx]:
                continue

            curr_mask_idxs = list(
                gidx_2_clipmaskidx[g_idx][clip_id]
            )

            cluster_mask_idxs.extend(curr_mask_idxs)

        if len(cluster_mask_idxs) == 0:
            continue

        mask_names = [
            fsam_mask_names[idx]
            for idx in cluster_mask_idxs
        ]

        num_figs = 0

        for start_idx in range(0, len(mask_names), imgs_per_fig):

            curr_mask_names = mask_names[
                start_idx:start_idx + imgs_per_fig
            ]

            fig, axes = plt.subplots(
                3,
                3,
                figsize=(12, 12)
            )

            axes = axes.flatten()

            for ax_idx, ax in enumerate(axes):

                if ax_idx >= len(curr_mask_names):
                    ax.axis("off")
                    continue

                mask_path = (
                    output_dir.parent.parent
                    / curr_mask_names[ax_idx]
                )

                img = plt.imread(mask_path)

                ax.imshow(img)

                global_mask_idx = start_idx + ax_idx

                ax.set_title(
                    f"{global_mask_idx}:"
                    f"{Path(mask_path).stem}",
                    fontsize=8
                )

                ax.axis("off")

            if cluster_label == -1:
                cluster_title = "Noise"
            else:
                cluster_title = f"Cluster {cluster_label}"

            fig.suptitle(
                f"FastSAM Masks for "
                f"Global Index {g_idx} "
                f"- {cluster_title}",
                fontsize=18
            )

            fig.tight_layout()

            num_figs += 1

            if num_figs >= max_figs_per_cluster:
                break

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, required=True, help="Directory to load the saved msmap output files.")
    args = parser.parse_args()
    with open(args.params, "r") as f:
        cfg = yaml.safe_load(f)
    
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    
    records = []
    for ds in cfg["datasets"]:
        print(f"\nLoading {ds['dataset']}")
        
        itr = ds['itr']
        
        output_dir = Path(ds["output_dir"])
        # with open(output_dir / f"clip_segs_itr{itr}.pt", "rb") as f:
        #     clip_ids = torch.load(f)
        with open(output_dir / f"gidx2clipcounts_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_clipcounts = pkl.load(f)
        with open(output_dir / f"gidx2clipdists_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_clipdists = pkl.load(f)
        with open(output_dir / f"gidx2clipmaskidx_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_clipmaskidx = pkl.load(f)
        with open(output_dir / f"saved_fastsam_mask_names_itr{itr}.pkl", "rb") as f:
            fsam_mask_names = pkl.load(f)
        pc = np.load(ds["global_pc"]) # (num_pts,4)        
        with open(output_dir / f"gidx2outliercounts_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_outlier_counts = pkl.load(f)
        with open(output_dir / f"gidx2counts_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_total_counts = pkl.load(f)
        with open(output_dir / f"gidx2obsvmediandists_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_dists_from_med = pkl.load(f)
        with open(output_dir / f"gidx2obsvmeandists_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_dists_from_mean = pkl.load(f)
        with open(output_dir / f"gidx2obsvlidardists_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_dists_from_lidar = pkl.load(f)
        with open(output_dir / f"gidx2clustercounts_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_cluster_counts = pkl.load(f)
        with open(output_dir / f"gidx2obsvcounts_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_obsv_counts = pkl.load(f)
        with open(output_dir / f"gidx2outlierratios_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_outlier_ratio = pkl.load(f)   
        with open(output_dir / f"gidx2silscores_noterrain_dict_itr{itr}.pkl", "rb") as f:
            gidx_2_silhouette_scores = pkl.load(f)
        # with open(output_dir / f"gidx2persscores_noterrain_dict_itr{itr}.pkl", "rb") as f:
        #     gidx_2_persistance_scores = pkl.load(f)     
        
        
        records.append({
            "dataset": ds["dataset"],
            "output_dir": ds["output_dir"],
            "itr": itr,

            # Loaded tensors / arrays
            # "clip_ids": clip_ids,
            "pc": pc,

            # CLIP association dictionaries
            "gidx_2_clipcounts": gidx_2_clipcounts,
            "gidx_2_clipdists": gidx_2_clipdists,
            "gidx_2_clipmaskidx": gidx_2_clipmaskidx,

            # FastSAM metadata
            "fsam_mask_names": fsam_mask_names,

            # Observation / statistics dictionaries
            "gidx_2_outlier_counts": gidx_2_outlier_counts,
            "gidx_2_total_counts": gidx_2_total_counts,
            "gidx_2_dists_from_med": gidx_2_dists_from_med,
            "gidx_2_dists_from_mean": gidx_2_dists_from_mean,
            "gidx_2_dists_from_lidar": gidx_2_dists_from_lidar,
            "gidx_2_cluster_counts": gidx_2_cluster_counts,
            "gidx_2_obsv_counts": gidx_2_obsv_counts,
            "gidx_2_outlier_ratio": gidx_2_outlier_ratio,
            "gidx_2_silhouette_scores": gidx_2_silhouette_scores,
            # "gidx_2_persistant_scores": gidx_2_persistance_scores,
        })
        print("Finished\n")

    sorted_outlier_ratios_all_datasets = []
    for d in records:
        sorted_or_items = sorted(d["gidx_2_outlier_ratio"].items(), key=lambda x: x[1], reverse=False)
        sorted_idx = [x[0] for x in sorted_or_items]
        sorted_ratios = [x[1] for x in sorted_or_items]
        
        sorted_outlier_ratios_all_datasets.append(sorted_ratios)
        
    #########################################################
    ## Print out silhuoette scores and cluster persistance ##
    #########################################################
    for d in records:
        print("\nDataset",d["dataset"])
        s_scores = [s_score for s_score in d["gidx_2_silhouette_scores"].values()]
        print(f"Silhuoette Scores: average={np.mean(s_scores):.3f}, max={np.max(s_scores):.3f}, min={np.min(s_scores):.3f}")
        # p_scores = [np.mean(p_score) if p_score.shape[0] > 0 else 0.0 for p_score in d["gidx_2_persistant_scores"].values()]
        # print(f"Cluster Persistance Scores: average={np.mean(p_scores):.3f}, max={np.max(p_scores):.3f}, min={np.min(p_scores):.3f}")
    
    ###############################################
    ## Plot 1: Outlier Ratio Across All Datasets ##
    ###############################################
    plt.figure(figsize=(8,6))
    legend = []
    for d_idx, sorted_or in enumerate(sorted_outlier_ratios_all_datasets):
        x = np.linspace(0, 1, len(sorted_or))
        plt.plot(x, sorted_or)
        legend.append(records[d_idx]["dataset"])
    plt.ylabel("Outlier Ratio",fontsize=17)
    plt.xlabel("Ratio of Points Sorted by Outlier Ratio",fontsize=17)
    plt.title("Point Cloud Outlier Presence",fontsize=18)
    plt.tick_params(axis='both', labelsize=15)
    # plt.tick_params(axis='y', labelsize=15)
    # plt.xticks([])
    plt.grid(True)
    plt.legend(legend, fontsize=13)
    plt.tight_layout()
    plt.show()
    
    ###########################################
    ## Plot 2: Cluster Histogram per Dataset ##
    ###########################################
    max_cluster_counts = [max(d["gidx_2_cluster_counts"].values()) for d in records]
    bins = [np.arange(1, max_cluster_counts[d_idx] + 2) - 0.5 for d_idx in range(len(max_cluster_counts))]
    for d_idx, d in enumerate(records):
        cluster_counts = np.array(
            list(d["gidx_2_cluster_counts"].values())
        )
        # Optional: remove zeros if desired
        cluster_counts = cluster_counts[cluster_counts > 0]
        
        plt.figure(figsize=(8,6))
        plt.hist(
            cluster_counts,
            bins=bins[d_idx],
            alpha=0.75,
        )
        plt.xlabel("Cluster Count", fontsize=17)
        plt.ylabel("Number of Points", fontsize=17)
        plt.title(
            f"Distribution of Cluster Counts Per Point for {d['dataset']}",
            fontsize=18
        )
        plt.tick_params(axis='both', labelsize=15)
        plt.grid(True)
        # plt.legend(fontsize=13)
        plt.tight_layout()
        plt.show()