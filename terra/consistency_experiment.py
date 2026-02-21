import argparse
import yaml
import pickle as pkl
import numpy as np

import open3d as o3d
import networkx as nx

def get_place_node_associations(terra_1, t_1tog, terra_2, t_2tog):
    """Given two Terra graphs and a transform between them, find the place node associations."""
    place_nodes_1 = [n for n, d in terra_1.nodes(data=True) if d["level"] == 1]
    # place_subgraph_1 = terra_1.subgraph(place_nodes_1)
    
    place_nodes_2 = [n for n, d in terra_2.nodes(data=True) if d["level"] == 1]
    # place_subgraph_2 = terra_2.subgraph(place_nodes_2)

    # Extract the 3D positions of the place nodes
    pos_xy_1 = np.array([terra_1.nodes[n]["pos"] for n in place_nodes_1])
    pos_xy_2 = np.array([terra_2.nodes[n]["pos"] for n in place_nodes_2])

    # Apply the transform to positions_v2 to bring them into the same frame
    transformed_positions_1 = (t_1tog[:3, :3] @ pos_xy_1.T).T + t_1tog[:3, 3]
    transformed_positions_2 = (t_2tog[:3, :3] @ pos_xy_2.T).T + t_2tog[:3, 3]

    # Now we can find associations based on proximity (e.g., using a nearest neighbor search)
    associations = {}
    for i, pos_1 in enumerate(transformed_positions_1):
        distances = np.linalg.norm(transformed_positions_2 - pos_1, axis=1)
        closest_idx = np.argmin(distances)
        associations[place_nodes_1[i]] = place_nodes_2[closest_idx]

    return associations


def main(yaml_file):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    
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
    
    # Define place node associations
    associations_1to2_dict = get_place_node_associations(terra_v1, np.eye(4), terra_v2, transform_v2_to_v1)
    associations_1to3_dict = get_place_node_associations(terra_v1, np.eye(4), terra_v3, transform_v3_to_v1)
    associations_1to4_dict = get_place_node_associations(terra_v1, np.eye(4), terra_v4, transform_v4_to_v1)
    
    
    pass


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