from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import time 
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster
import networkx as nx

from utils import tensor_cosine_similarity
from terra_utils import load_terra

def plot_merge_diagnostics(model, num_cuts = 4):
    """
    Plots:
    1) Merge distances
    2) First difference of merge distances
    3) Second difference of merge distances
    """

    distances = model.distances_

    # Sort just in case (they should already be sorted for agglomerative)
    distances = np.sort(distances)

    # First derivative
    delta = np.diff(distances)

    # Second derivative
    delta2 = np.diff(delta)

    # ---- Plot 1: Raw merge distances ----
    plt.figure()
    plt.plot(distances)
    plt.title("Merge Distance vs Merge Index")
    plt.xlabel("Merge Index")
    plt.ylabel("Distance")
    plt.show()

    # # ---- Plot 2: First derivative ----
    # plt.figure()
    # plt.plot(delta)
    # plt.title("First Difference of Merge Distances")
    # plt.xlabel("Merge Index")
    # plt.ylabel("Δ Distance")
    # plt.show()

    # # ---- Plot 3: Second derivative ----
    # plt.figure()
    # plt.plot(delta2)
    # plt.title("Second Difference of Merge Distances")
    # plt.xlabel("Merge Index")
    # plt.ylabel("Δ² Distance")
    # plt.show()

    # ---- Print suggested cut points ----
    topN = np.argsort(delta)[-num_cuts:]  # largest num_cuts jumps
    # topN = np.argsort(delta2)[-num_cuts:]
    topN = np.sort(topN)

    print(f"\nSuggested cut indices (largest {num_cuts} jumps):", topN)
    print("Suggested cut distances based on Delta:", distances[topN])
    # print("Suggested cut distances based on Delta2:", distances[topN])
    return distances[topN]

def plot_dendrogram(model, cut_heights=None, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    if cut_heights is not None:
        for h in cut_heights:
            plt.axhline(y=h, color='k', linestyle='--')
    
def edge_weight(G, n1_id, n2_id, beta, gamma):  
    cos_sim_places = tensor_cosine_similarity(G.nodes[n1_id]["embedding"], G.nodes[n2_id]["embedding"]).item()
    
    euclidean_dist = np.linalg.norm(G.nodes[n1_id]["pos"] - G.nodes[n2_id]["pos"]) 
    
    weight = beta * euclidean_dist + gamma * (1 - min(cos_sim_places,1))
    
    return weight

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--Terra', type=str, help="Terra class object filepath")
    parser.add_argument('--beta', type=float)
    parser.add_argument('--gamma', type=float)
    args = parser.parse_args()
    terra = load_terra(args.Terra)
    
    place_nodes = [n for n, d in terra.terra_3dsg.nodes(data=True) if d['level'] == 1]
    places_graph = terra.terra_3dsg.subgraph(place_nodes)
    
    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=50, metric='precomputed', linkage='average')
    
    terra_graph_dense = places_graph.copy()

    t0 = time.time()
    for n1_id in tqdm(list(places_graph.nodes), desc="Building dense graph"):
        for n2_id in list(places_graph.nodes):
            if n1_id == n2_id:
                continue
            if not terra_graph_dense.has_edge(n1_id, n2_id):
                wij = edge_weight(terra_graph_dense, n1_id, n2_id, args.beta, args.gamma)
                terra_graph_dense.add_edge(n1_id, n2_id, weight=wij)
    t1 = time.time()
    print(f"Finished computing dense graph edge weights in {t1-t0:.2f} seconds")
    
    dist_matrix = nx.to_numpy_array(terra_graph_dense, weight='weight')
    agg_model = ac.fit(dist_matrix)
    
    cut_heights = plot_merge_diagnostics(agg_model, num_cuts=5)#4)
    
    plt.title("Dendrogram for Agglomerative Clustering of GVD Nodes")
    plot_dendrogram(agg_model, cut_heights=cut_heights)
    plt.xlabel("Node Index")
    plt.ylabel("Distance")
    plt.show()