import argparse
from sympy import Number
import yaml
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

import open3d as o3d
import networkx as nx

from visualize_terra import TerraVisualizer
from utils import tensor_cosine_similarity

def invert_transform(T):
    """Invert a 4x4 homogeneous transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def transform_graph_positions(G, T):
    """Apply a homogeneous transformation to the 'pos' attribute of each node in the graph."""
    transformed_G = G.copy()
    for n, d in transformed_G.nodes(data=True):
        if d.get("pos") is None:
            continue  # Skip nodes without a 'pos' attribute
        pos_xy = np.array(d["pos"])
        pos_xyz = np.append(pos_xy, 0)  # Add z=0 to make it 3D
        # Convert to homogeneous coordinates
        pos_homogeneous = np.append(pos_xyz, 1)
        # Apply the transformation
        transformed_pos_homogeneous = T @ pos_homogeneous
        # Convert back to 3D coordinates
        transformed_pos = transformed_pos_homogeneous[:3]
        # Update the node's position
        transformed_G.nodes[n]["pos"] = transformed_pos[:2]  # Keep only x and y for the graph
    return transformed_G

def display_aligned_place_nodes(terra_1, terra_2, terra_3, terra_4):
    vis = TerraVisualizer(level_offset=50, num_terrains=7)
    geo = []
    geo.extend(vis.display_places(terra_1, return_geo=True))
    geo.extend(vis.display_places(terra_2, return_geo=True))
    geo.extend(vis.display_places(terra_3, return_geo=True))
    geo.extend(vis.display_places(terra_4, return_geo=True))
    o3d.visualization.draw_geometries(geo)

def get_viridis_divergent_cmap():
    # Get viridis colormap
    viridis = cm.get_cmap('viridis')

    # Sample it
    colors = viridis(np.linspace(0, 1, 128))

    # Create symmetric version around yellow (center)
    left = colors
    right = colors[::-1]

    new_colors = np.vstack((left, right))

    viridis_div = mcolors.LinearSegmentedColormap.from_list(
        'viridis_diverging',
        new_colors
    )
    return viridis_div
 
def plot_heatmap(matrix, title="Confusion Matrix", labels=None, cmap_min=None, cmap_max=None):
    """
    Plot a heatmap of a distance matrix with values shown in each cell.

    Args:
        matrix (list of lists or np.ndarray): NxN matrix
        title (str): Plot title
        labels (list of str): Optional axis labels
    """
    matrix = np.array(matrix)
    n = matrix.shape[0]

    fig, ax = plt.subplots()

    # Show heatmap
    flipped = False
    if cmap_min is not None and cmap_max is not None:
        flipped = True
        im = ax.imshow(matrix, cmap='viridis_r', interpolation="nearest", vmin=cmap_min, vmax=cmap_max)
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=15)
        # cbar.set_label("Distance", fontsize=15)
    elif matrix.max() > 1.01 or matrix.min() < 0.0:
        max_dev = max(abs(matrix.min() - 1), abs(matrix.max() - 1))
        norm = mcolors.TwoSlopeNorm(
            vmin=0.0, #1 - max_dev,
            vcenter=1.0,
            vmax=2.0, #1 + max_dev
        )
        im = ax.imshow(matrix, cmap=get_viridis_divergent_cmap(), interpolation="nearest", norm=norm)
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks(np.arange(0.0, 2.0 + 0.25, 0.25))
        cbar.ax.tick_params(labelsize=15)
        # cbar.set_label("Distance", fontsize=15)
    else:
        cmap_min = 0.0
        cmap_max = 1.0
        im = ax.imshow(matrix, interpolation="nearest", vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=15)
        # cbar.set_label("Distance", fontsize=15)

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    # Rotate x labels if present
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            if flipped:
                text = ax.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    # color="black" if value > (np.nanmin(matrix) + (np.nanmax(matrix) - np.nanmin(matrix))/2) else "white"
                    color="black" if value <= (cmap_min + (cmap_max - cmap_min)/2) else "white"
                )
            elif matrix.max() > 1.0 or matrix.min() < 0.0:
                text = ax.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="black"
                )
            else:
                text = ax.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    # color="black" if value > (np.nanmin(matrix) + (np.nanmax(matrix) - np.nanmin(matrix))/2) else "white"
                    color="black" if value > (cmap_min + (cmap_max - cmap_min)/2) else "white"
                )

    ax.set_title(title, fontsize=18)
    # ax.set_xlabel("Graph index", fontsize=17)
    # ax.set_ylabel("Graph index", fontsize=17)
    plt.tick_params(axis='both', labelsize=15)

    plt.tight_layout()
    plt.show()
    
def load_and_transform_terra_graphs(cfg, display=False):
    # Load nx.Graphs and transforms
    with open(cfg["terra_v1"], "rb") as f:
        terra_v1 = pkl.load(f)
    with open(cfg["terra_v2"], "rb") as f:
        terra_v2 = pkl.load(f)
    with open(cfg["terra_v3"], "rb") as f:
        terra_v3 = pkl.load(f)
    with open(cfg["terra_v4"], "rb") as f:
        terra_v4 = pkl.load(f)
    
    transform_v2_to_v1 = np.load(cfg["transform_v2_to_v1"])
    transform_v3_to_v1 = np.load(cfg["transform_v3_to_v1"])
    transform_v4_to_v1 = np.load(cfg["transform_v4_to_v1"])
    
    # Transform all graphs to the same frame (e.g., v1 frame)
    terra_v1_aligned = terra_v1
    terra_v2_aligned = transform_graph_positions(terra_v2, transform_v2_to_v1)
    terra_v3_aligned = transform_graph_positions(terra_v3, transform_v3_to_v1)
    terra_v4_aligned = transform_graph_positions(terra_v4, transform_v4_to_v1)
    
    if display:
        print("Before applying transformations")
        display_aligned_place_nodes(terra_v1, terra_v2, terra_v3, terra_v4)
        print("After applying transformations")
        display_aligned_place_nodes(terra_v1_aligned, terra_v2_aligned, terra_v3_aligned, terra_v4_aligned)
        
    return terra_v1_aligned, terra_v2_aligned, terra_v3_aligned, terra_v4_aligned
    

def associate_place_nodes_1_to_2(terra_1, terra_2):
    """Given two Terra graphs, find the place node associations."""
    place_nodes_1 = [n for n, d in terra_1.nodes(data=True) if d["level"] == 1]
    # place_subgraph_1 = terra_1.subgraph(place_nodes_1)
    
    place_nodes_2 = [n for n, d in terra_2.nodes(data=True) if d["level"] == 1]
    # place_subgraph_2 = terra_2.subgraph(place_nodes_2)

    # Extract the 3D positions of the place nodes
    pos_xy_1 = np.array([terra_1.nodes[n]["pos"] for n in place_nodes_1])
    pos_xy_2 = np.array([terra_2.nodes[n]["pos"] for n in place_nodes_2])

    # Now we can find associations based on proximity (e.g., using a nearest neighbor search)
    associations = {}
    for i, pos_1 in enumerate(pos_xy_1):
        distances = np.linalg.norm(pos_xy_2 - pos_1, axis=1)
        closest_idx = np.argmin(distances)
        associations[place_nodes_1[i]] = place_nodes_2[closest_idx]

    return associations  

def associate_region_nodes_1_to_2(terra_1, terra_2):
    """Given two Terra graphs, find the place node associations."""
    max_level_1 = 0
    region_nodes_1 = {}
    for n, d in terra_1.nodes(data=True):
        if max_level_1 < d["level"]:
            max_level_1 = d["level"]
        if d["level"] > 1:
            if d["level"] not in region_nodes_1:
                region_nodes_1[d["level"]] = []
            region_nodes_1[d["level"]].append(n)
    # print("Max level in terra 1:", max_level_1)
        
    max_level_2 = 0
    region_nodes_2 = {}
    for n, d in terra_2.nodes(data=True):
        if max_level_2 < d["level"]:
            max_level_2 = d["level"]
        if d["level"] > 1:
            if d["level"] not in region_nodes_2:
                region_nodes_2[d["level"]] = []
            region_nodes_2[d["level"]].append(n)
    # print("Max level in terra 2:", max_level_2)
    
    if max_level_1 != max_level_2:
        print("WARNING: Graphs have different max levels, associations may not be reliable.")
    
    # Now we can find associations based on proximity and matching level (e.g., using a nearest neighbor search)
    associations = {}
    for lvl1, n1_list in region_nodes_1.items():
        for lvl2, n2_list in region_nodes_2.items():
            if lvl1 == lvl2:
                pos_xy_1 = np.array([terra_1.nodes[n]["pos"] for n in n1_list])
                pos_xy_2 = np.array([terra_2.nodes[n]["pos"] for n in n2_list])
                
                for i, pos_1 in enumerate(pos_xy_1):                    
                    distances = np.linalg.norm(pos_xy_2 - pos_1, axis=1)
                    closest_idx = np.argmin(distances)
                    associations[n1_list[i]] = n2_list[closest_idx]
    
    return associations

def select_place_nodes_by_region(terra, region_node):
    selected_place_nodes = set()
    queue = [region_node]
    while queue:
        node = queue.pop()
        node_level = terra.nodes[node]["level"]
        for nbr in terra.neighbors(node):
            nbr_level = terra.nodes[nbr]["level"]
            if nbr_level == 1: # children are places, stop descent
                selected_place_nodes.add(nbr)
            elif nbr_level < node_level: # children are still regions, keep descending
                queue.append(nbr)            
    return selected_place_nodes

def count_associated_place_nodes(place_nodes_1, place_nodes_2, place_associations):
    count = 0
    for k, v in place_associations.items():
        if k in place_nodes_1 and v in place_nodes_2:
            count += 1
    return count


def graph_consistency_eval(terras, place_associations, region_associations):
    """Check consistency of graph structures across terrains."""
    # Check if the number of place nodes is consistent
    num_place_nodes = [len([n for n, d in t.nodes(data=True) if d["level"] == 1]) for t in terras]
    print("Number of place nodes in each graph:", num_place_nodes)
    n = len(terras)
    ratio_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            ratio_matrix[i, j] = num_place_nodes[i] / num_place_nodes[j]
    plot_heatmap(
        ratio_matrix, 
        title="Place Node Count Ratios", 
        labels=[f"V{i+1}" for i in range(n)],
    )
    
    # # Check degree consistency
    # distances = []
    # for i in range(len(place_associations)):
    #     d_i = []
    #     for j in range(len(place_associations[i])):
    #         d_i2j = []
    #         for k, v in place_associations[i][j].items():
    #             deg_k = terras[i].degree(k)
    #             deg_v = terras[j].degree(v)
    #             # print(f"Graph {i+1} node {k} degree: {deg_k}, Graph {j+1} node {v} degree: {deg_v}, DIFF: {np.abs(deg_k - deg_v)}")
    #             d_i2j.append(np.abs(deg_k - deg_v))
    #         d_i.append(np.mean(d_i2j))
    #     distances.append(d_i)
    # plot_heatmap(distances, title="Mean Degree Difference", labels=[f"V{i+1}" for i in range(len(place_associations))])
    
    # Check ratio of children place nodes of associated region nodes
    num_region_nodes = [len([n for n, d in t.nodes(data=True) if d["level"] > 1]) for t in terras]
    print("Number of region nodes in each graph:", num_region_nodes)
    n = len(terras)
    region_ratio_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            ratios = []
            for k, v in region_associations[i][j].items():
                place_nodes_1 = select_place_nodes_by_region(terras[i], k)
                place_nodes_2 = select_place_nodes_by_region(terras[j], v)
                matching_associations = count_associated_place_nodes(place_nodes_1, place_nodes_2, place_associations[i][j])
                ratio = matching_associations / len(place_nodes_1) if len(place_nodes_1) > 0 else 0
                ratios.append(ratio)
            region_ratio_matrix[i, j] = np.mean(ratios)
    plot_heatmap(
        region_ratio_matrix, 
        title="Place Node Clustering Consistency", 
        labels=[f"V{i+1}" for i in range(n)],
    )


def geometric_consistency_eval(terras, associations):
    """Check consistency of geometric properties across place nodes."""
    # Check if the distances between associated nodes are consistent
    distances = []
    for i in range(len(associations)):
        d_i = []
        for j in range(len(associations[i])):
            d_i2j = []
            for k, v in associations[i][j].items():
                pos_k = terras[i].nodes[k]["pos"]
                pos_v = terras[j].nodes[v]["pos"]
                distance = np.linalg.norm(np.array(pos_k) - np.array(pos_v))
                d_i2j.append(distance)
            d_i.append(np.mean(d_i2j))
        distances.append(d_i)
    plot_heatmap(
        distances, 
        title="Mean Place Node Distance [m]", 
        labels=[f"V{i+1}" for i in range(len(associations))],
        cmap_min=np.nanmin(distances),
        cmap_max=5.0 # max distance between nodes
    )


def semantic_consistency_eval(terras, associations):
    # Mean cosine_similarity scores across all associations
    scores = []
    for i in range(len(associations)):
        s_i = []
        for j in range(len(associations[i])):
            s_i2j = []
            for k, v in associations[i][j].items():
                embedding_k = terras[i].nodes[k]["embedding"]
                embedding_v = terras[j].nodes[v]["embedding"]
                cos_sim = tensor_cosine_similarity(embedding_k, embedding_v)
                s_i2j.append(cos_sim.item())
            s_i.append(np.mean(s_i2j))
        scores.append(s_i)
    plot_heatmap(
        scores, 
        title="Mean Place Node Cosine Similarity", 
        labels=[f"V{i+1}" for i in range(len(associations))],
    )
    
    # Number of matching terrain IDs
    matches = []
    for i in range(len(associations)):
        m_i = []
        for j in range(len(associations[i])):
            m_i2j = []
            for k, v in associations[i][j].items():
                terrain_id_k = terras[i].nodes[k]["terrain_id"]
                terrain_id_v = terras[j].nodes[v]["terrain_id"]
                match = int(terrain_id_k == terrain_id_v)
                m_i2j.append(match)
            m_i.append(np.mean(m_i2j))
        matches.append(m_i)
    plot_heatmap(
        matches, 
        title="Terrain Match Consistency", 
        labels=[f"V{i+1}" for i in range(len(associations))]
    )


def main(yaml_file):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    # print(cfg)
    
    tv1, tv2, tv3, tv4 = load_and_transform_terra_graphs(cfg, display=False)
    terras = [tv1, tv2, tv3, tv4]
    
    # Define place node associations
    place_associations = []
    region_associations = []
    for i in range(4):
        place_row = []
        region_row = []
        for j in range(4):
            print("Associating graphs v{} and v{}".format(i+1, j+1))
            place_row.append(associate_place_nodes_1_to_2(terras[i], terras[j]))
            region_row.append(associate_region_nodes_1_to_2(terras[i], terras[j]))
        place_associations.append(place_row)
        region_associations.append(region_row)
    
    graph_consistency_eval(terras, place_associations, region_associations)
    geometric_consistency_eval(terras, place_associations)
    semantic_consistency_eval(terras, place_associations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="YAML file describing Terra datasets"
    )
    args = parser.parse_args()

    main(args.params)