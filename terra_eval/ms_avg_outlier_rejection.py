from argparse import ArgumentParser
import pickle as pkl
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

from terra.utils import tensor_cosine_similarity, load_terra

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

def compute_weighted_geometric_median(X, w, max_iter=100, tol=1e-6):
    # Initialize with weighted mean
    best_x = (X * w.unsqueeze(1)).sum(dim=0) / w.sum()
    for i in range(max_iter):
        # Compute distances from current point to all points
        # diff = X - best_x.unsqueeze(0)
        # dists = diff.norm(dim=1).clamp(min=1e-12)
        diff = tensor_cosine_similarity(X, best_x.unsqueeze(0)) # [num_clip_ids,]
        dist = 1 - diff # convert cosine similarity to distance
        
        # Compute weights inversely proportional to distances
        weights = w / (dist.squeeze() + 1e-12) # avoid division by zero
        
        # Update current point using weighted average of all points with new weights
        new_x = (X * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

        if (new_x - best_x).norm() < tol:
            print("Converged after {} iterations".format(i+1))
            break
        best_x = new_x
    return best_x

def count_outliers(X, w, threshold=0.3):
    # Normalize embeddings
    X = X / X.norm(dim=1, keepdim=True)

    # Weighted mean
    mu = (X * w.unsqueeze(1)).sum(dim=0) / w.sum()
    mu = mu / mu.norm()

    # Cosine distance
    sim = tensor_cosine_similarity(X, mu.unsqueeze(0))
    dist = 1 - sim

    # Count outliers
    return (dist > threshold).sum().item()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--Terra', type=str, help="Terra class object filepath")
    parser.add_argument('--terrain_gidx', type=str, help="List of global indices assigned to terrain (e.g. terrain_gidx.pkl)")
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
    
    # # skip_count = 5000
    # # count = 0
    # for g_idx, cid_2_cnt in tqdm(gidx_2_clipcounts.items(),desc="Selecting point with most outliers"):
    #     if g_idx not in terrain_gidxs:
    #     # if g_idx in terrain_gidxs:
    #         continue
    #     # if count < skip_count:
    #     #     count += 1
    #     #     continue
    #     # chosen_gidx = g_idx
    #     # clipid_2_counts = cid_2_cnt
    #     # break
        
    #     # Build X and w
    #     X = torch.zeros((len(cid_2_cnt), 512), device=device)
    #     w = torch.zeros((len(cid_2_cnt),), device=device)

    #     for i, (clip_id, count) in enumerate(cid_2_cnt.items()):
    #         X[i,:] = clip_ids[clip_id,:]
    #         w[i] = count

    #     num_outliers = count_outliers(X, w, threshold=0.3)

    #     if num_outliers > max_outliers and len(cid_2_cnt) > 30:
    #         max_outliers = num_outliers
    #         chosen_gidx = g_idx
    #         clipid_2_counts = cid_2_cnt
    #         print("Number of clip detections for global index {}: {}".format(g_idx, w.sum().item()))
    #         print("New max number of outliers",max_outliers)
    #         break
    
    ## Random gidx
    random_key = random.choice(list(gidx_2_clipcounts.keys()))
    chosen_gidx = random_key
    clipid_2_counts = gidx_2_clipcounts[chosen_gidx]
    # while len(clipid_2_counts) < 30:
    while len(clipid_2_counts) < 30 or chosen_gidx not in terrain_gidxs:
        random_key = random.choice(list(gidx_2_clipcounts.keys()))
        chosen_gidx = random_key
        clipid_2_counts = gidx_2_clipcounts[chosen_gidx]
    # Build X and w
    X = torch.zeros((len(clipid_2_counts), 512), device=device)
    w = torch.zeros((len(clipid_2_counts),), device=device)
    for i, (clip_id, count) in enumerate(clipid_2_counts.items()):
        X[i,:] = clip_ids[clip_id,:]
        w[i] = count
    max_outliers = count_outliers(X, w, threshold=0.3)
    print("Number of clip detections for global index {}: {}".format(chosen_gidx, w.sum().item()))
    
    
    print(f"Chosen gidx: {chosen_gidx}")
    print(f"Number of outliers: {max_outliers}")
        
    
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
    trimmed_mean_w_02 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.2)
    trimmed_mean_w_04 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.4)
    trimmed_mean_w_06 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.6)
    trimmed_mean_w_09 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.9)
    trimmed_mean_w_095 = compute_weighted_trimmed_mean(X, w, dist, trim_percent=0.95)
    geometric_median_w = compute_weighted_geometric_median(X, w, max_iter=100, tol=1e-5)
    max_w = X[w.argmax(),:]
    
    mean_xy = np.array([mu_w[92].item(), mu_w[133].item()])
    medoid_xy = np.array([medoid_w[92].item(), medoid_w[133].item()])
    trimmed_mean_02_xy = np.array([trimmed_mean_w_02[92].item(), trimmed_mean_w_02[133].item()])
    trimmed_mean_04_xy = np.array([trimmed_mean_w_04[92].item(), trimmed_mean_w_04[133].item()])
    trimmed_mean_06_xy = np.array([trimmed_mean_w_06[92].item(), trimmed_mean_w_06[133].item()])
    trimmed_mean_09_xy = np.array([trimmed_mean_w_09[92].item(), trimmed_mean_w_09[133].item()])
    trimmed_mean_095_xy = np.array([trimmed_mean_w_095[92].item(), trimmed_mean_w_095[133].item()])
    has_trimmed_mean_095_nan = np.isnan(trimmed_mean_095_xy).any()
    if has_trimmed_mean_095_nan:
        print("Warning: Weighted Trimmed Mean t=0.95 contains NaNs; skipping it in plots.")
    geometric_median_xy = np.array([geometric_median_w[92].item(), geometric_median_w[133].item()])
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
    if not has_trimmed_mean_095_nan:
        plt.scatter(trimmed_mean_095_xy[0], trimmed_mean_095_xy[1], color='black', label='Weighted Trimmed Mean t=0.95', marker='*', s=200)
    plt.xlabel("Embedding Dimension 92")
    plt.ylabel("Embedding Dimension 133")
    if is_terrain:
        plt.title(f"Clip Embeddings and Averages for Terrain Global Index {chosen_gidx}")
    else:
        plt.title(f"Clip Embeddings and Averages for Nonterrain Global Index {chosen_gidx}")
    plt.legend()
    plt.show()

    # t-SNE projection for checking potential multimodality
    if X.shape[0] >= 3:
        x_np = X.detach().cpu().numpy()
        
        if has_trimmed_mean_095_nan:
            learned_embs = torch.stack([
                mu_w,
                medoid_w,
                max_w,
                trimmed_mean_w_02,
                trimmed_mean_w_04,
                trimmed_mean_w_06,
                trimmed_mean_w_09,
            ], dim=0).detach().cpu().numpy()

            learned_names = [
                "Weighted Mean",
                "Weighted Medoid",
                "Max Embedding",
                "Weighted Trimmed Mean t=0.2",
                "Weighted Trimmed Mean t=0.4",
                "Weighted Trimmed Mean t=0.6",
                "Weighted Trimmed Mean t=0.9",
            ]
            learned_markers = ["X", "^", "P", "*", "*", "*", "*"]
            learned_colors = ["red", "green", "brown", "orange", "magenta", "green", "blue"]
        else:
            learned_embs = torch.stack([
                mu_w,
                medoid_w,
                max_w,
                trimmed_mean_w_02,
                trimmed_mean_w_04,
                trimmed_mean_w_06,
                trimmed_mean_w_09,
                trimmed_mean_w_095,
            ], dim=0).detach().cpu().numpy()

            learned_names = [
                "Weighted Mean",
                "Weighted Medoid",
                "Max Embedding",
                "Weighted Trimmed Mean t=0.2",
                "Weighted Trimmed Mean t=0.4",
                "Weighted Trimmed Mean t=0.6",
                "Weighted Trimmed Mean t=0.9",
                "Weighted Trimmed Mean t=0.95",
            ]
            learned_markers = ["X", "^", "P", "*", "*", "*", "*", "*"]
            learned_colors = ["red", "green", "brown", "orange", "magenta", "green", "blue", "black"]

        tsne_input = np.vstack([x_np, learned_embs])
        n_samples = x_np.shape[0]
        print(n_samples)
        perplexity = min(30, n_samples - 1)
        perplexity = max(2, perplexity)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42
        )
        tsne_all = tsne.fit_transform(tsne_input)
        tsne_xy = tsne_all[:n_samples]
        tsne_learned = tsne_all[n_samples:]

        plt.figure()
        plt.scatter(
            tsne_xy[:, 0],
            tsne_xy[:, 1],
            c=w_np,
            s=sizes,
            cmap="viridis",
            alpha=0.75,
            label="Clip Embeddings"
        )

        for i, name in enumerate(learned_names):
            plt.scatter(
                tsne_learned[i, 0],
                tsne_learned[i, 1],
                color=learned_colors[i],
                marker=learned_markers[i],
                s=220,
                edgecolors="white",
                linewidths=0.7,
                label=name
            )

        cbar = plt.colorbar()
        cbar.set_label("Detection Count (weight)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        if is_terrain:
            plt.title(f"t-SNE of Clip Embeddings for Terrain Global Index {chosen_gidx}")
        else:
            plt.title(f"t-SNE of Clip Embeddings for Nonterrain Global Index {chosen_gidx}")
        plt.legend(fontsize=8, loc="best")
        plt.show()
    else:
        print(f"Skipping t-SNE: need at least 3 points, got {X.shape[0]}")

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