#!/usr/bin/env python3

import yaml
import pickle as pkl
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import networkx as nx

from terra_utils import load_terra


def terra_size_mb(terra_obj):
    """Approximate in-memory size of a Terra object in MB."""
    return len(pkl.dumps(terra_obj, protocol=pkl.HIGHEST_PROTOCOL)) / (1024 ** 2)


# def make_combined_legend(color_by_dataset):
    # """Create a compact legend with a divider between camera shapes and dataset colors."""

    # legend_handles = []

    # # # -----------------------------
    # # # Cameras (shape)
    # # # -----------------------------
    # # legend_handles.extend([
    # #     Line2D([0], [0], marker='o', color='k', linestyle='None',
    # #            markersize=8, label='1 camera'),
    # #     Line2D([0], [0], marker='^', color='k', linestyle='None',
    # #            markersize=8, label='3 cameras'),
    # # ])

    # # # -----------------------------
    # # # Divider line
    # # # -----------------------------
    # # legend_handles.append(Line2D([0], [0], color='k', linestyle='-', linewidth=1.2, label=''))

    # # -----------------------------
    # # Dataset (color)
    # # -----------------------------
    # legend_handles.extend([
    #     Line2D([0], [0], marker='o', color=color, linestyle='None',
    #            markersize=8, label=dataset)
    #     for dataset, color in color_by_dataset.items()
    # ])

    # return legend_handles
    
def make_combined_legend(color_by_dataset, marker_by_dataset):
    """Legend showing both marker and color per dataset."""

    legend_handles = []

    for dataset in color_by_dataset.keys():
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=marker_by_dataset[dataset],
                color=color_by_dataset[dataset],
                linestyle='None',
                markersize=8,
                label=dataset
            )
        )

    return legend_handles



# def scatter_plot(records, color_by_dataset, marker_by_cameras,
#                  legend_handles,
#                  x_key, y_key, xlabel, ylabel, title):
def scatter_plot(records, color_by_dataset, marker_by_dataset,
                 legend_handles,
                 x_key, y_key, xlabel, ylabel, title):

    plt.figure(figsize=(8.5, 5))

    for r in records:
        color = color_by_dataset[r["dataset"]]
        # marker = marker_by_cameras[r["cameras"]]
        marker = marker_by_dataset[r["dataset"]]

        plt.scatter(
            r[x_key],
            r[y_key],
            marker=marker,
            s=90,
            color=color,
            linewidths=1.8,
        )

    plt.xlabel(xlabel, fontsize=17)
    plt.ylabel(ylabel, fontsize=17)
    plt.title(title, fontsize=18)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=15)

    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=15,
        frameon=True,
    )

    plt.tight_layout()
    
# def two_panel_traj_plot_vertical(records, color_by_dataset, marker_by_cameras,
#                                  legend_handles):
def two_panel_traj_plot_vertical(records, color_by_dataset, marker_by_dataset,
                                 legend_handles):
    """
    Two subplots stacked vertically:
      (top) Memory vs trajectory length
      (bottom) Nodes vs trajectory length
    Single legend in the top plot (lower right).
    """

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # -----------------------------
    # Top plot: Memory vs Trajectory
    # -----------------------------
    ax = axes[0]
    for r in records:
        color = color_by_dataset[r["dataset"]]
        # marker = marker_by_cameras[r["cameras"]]
        marker = marker_by_dataset[r["dataset"]]

        ax.scatter(
            r["traj_len"],
            r["memory_mb"],
            marker=marker,
            s=90,
            edgecolors=color,
            facecolors=color,
            linewidths=1.8,
        )

    ax.set_ylabel("Memory Size (MB)", fontsize=17)
    ax.set_title("Terra Memory vs Trajectory Length", fontsize=18)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=15)

    # Place legend in the lower right inside the top subplot
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=13,
        frameon=True,
    )

    # -----------------------------
    # Bottom plot: Nodes vs Trajectory
    # -----------------------------
    ax = axes[1]
    for r in records:
        color = color_by_dataset[r["dataset"]]
        # marker = marker_by_cameras[r["cameras"]]
        marker = marker_by_dataset[r["dataset"]]

        ax.scatter(
            r["traj_len"],
            r["nodes"],
            marker=marker,
            s=90,
            edgecolors=color,
            facecolors=color,
            linewidths=1.8,
        )

    ax.set_xlabel("Trajectory Length (m)", fontsize=17)
    ax.set_ylabel("Number of 3DSG Nodes", fontsize=17)
    ax.set_title("3DSG Nodes vs Trajectory Length", fontsize=18)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=15)

    # -----------------------------
    # Adjust layout
    # -----------------------------
    plt.tight_layout()

# def two_panel_traj_plot_vertical(records, color_by_dataset, marker_by_cameras,
#                                  legend_handles):
def two_panel_area_plot_vertical(records, color_by_dataset, marker_by_dataset,
                                 legend_handles):
    """
    Two subplots stacked vertically:
      (top) Memory vs trajectory length
      (bottom) Nodes vs trajectory length
    Single legend in the top plot (lower right).
    """

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # -----------------------------
    # Top plot: Memory vs Trajectory
    # -----------------------------
    ax = axes[0]
    for r in records:
        color = color_by_dataset[r["dataset"]]
        # marker = marker_by_cameras[r["cameras"]]
        marker = marker_by_dataset[r["dataset"]]

        ax.scatter(
            r["area_m2"],
            r["memory_mb"],
            marker=marker,
            s=90,
            edgecolors=color,
            facecolors=color,
            linewidths=1.8,
        )

    ax.set_ylabel("Memory Size (MB)", fontsize=17)
    ax.set_title("Terra Memory vs Area", fontsize=18)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=15)

    # Place legend in the lower right inside the top subplot
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=13,
        frameon=True,
    )

    # -----------------------------
    # Bottom plot: Nodes vs Trajectory
    # -----------------------------
    ax = axes[1]
    for r in records:
        color = color_by_dataset[r["dataset"]]
        # marker = marker_by_cameras[r["cameras"]]
        marker = marker_by_dataset[r["dataset"]]

        ax.scatter(
            r["area_m2"],
            r["nodes"],
            marker=marker,
            s=90,
            edgecolors=color,
            facecolors=color,
            linewidths=1.8,
        )

    ax.set_xlabel("Area (m²)", fontsize=17)
    ax.set_ylabel("Number of 3DSG Nodes", fontsize=17)
    ax.set_title("3DSG Nodes vs Area", fontsize=18)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=15)

    # -----------------------------
    # Adjust layout
    # -----------------------------
    plt.tight_layout()

def compute_square_area(G):
    place_nodes = [n for n, d in G.nodes(data=True) if d["level"] == 1]
    pg = G.subgraph(place_nodes)
    pos = nx.get_node_attributes(pg, 'pos')
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    width_m = max(x_vals) - min(x_vals) # [m]
    height_m = max(y_vals) - min(y_vals) # [m]
    return width_m * height_m
    # width_km = width / 1000
    # height_km = height / 1000
    # return width_km * height_km
    
def compute_pc_area(pc, resolution=0.4):
    xy = pc[:,:2]
    grid_indices = np.floor(xy / resolution).astype(int)
    unique_cells = np.unique(grid_indices, axis=0)
    num_voxels = unique_cells.shape[0]
    area = num_voxels * (resolution ** 2) # [m^2]
    return area
    

def main(yaml_file):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

    # -----------------------------
    # Visual encodings
    # -----------------------------
    marker_by_cameras = {
        1: "o",
        3: "^",
    }
    marker_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]

    datasets = sorted({ds["dataset"] for ds in cfg["datasets"]})
    cmap = plt.get_cmap("tab10")
    color_by_dataset = {}
    marker_by_dataset = {}

    for i, d in enumerate(datasets):
        color_by_dataset[d] = cmap(i % cmap.N)
        marker_by_dataset[d] = marker_cycle[i % len(marker_cycle)]

    # -----------------------------
    # Load Terra data
    # -----------------------------
    records = []

    for ds in cfg["datasets"]:
        print(f"Loading {ds['dataset']} | {ds['cameras']} cam")

        terra = load_terra(ds["terra_path"])
        # area = compute_square_area(terra.terra_3dsg)
        area = compute_pc_area(terra.pc, resolution=0.4)
        
        records.append({
            "dataset": ds["dataset"],
            "cameras": ds["cameras"],
            "traj_len": ds["trajectory_length_m"],
            "area_m2": area,
            "memory_mb": terra_size_mb(terra),
            "nodes": terra.terra_3dsg.number_of_nodes(),
        })
    
    # Print records
    for r in records:
        print(r)
    
    # -----------------------------
    # Build legend once
    # -----------------------------
    legend_handles = make_combined_legend(color_by_dataset, marker_by_dataset)

    # -----------------------------
    # Plots
    # -----------------------------
    scatter_plot(
        records,
        color_by_dataset,
        # marker_by_cameras,
        marker_by_dataset,
        legend_handles,
        x_key="traj_len",
        y_key="memory_mb",
        xlabel="Trajectory Length (m)",
        ylabel="Memory Size (MB)",
        title="Terra Memory vs Trajectory Length",
    )

    scatter_plot(
        records,
        color_by_dataset,
        # marker_by_cameras,
        marker_by_dataset,
        legend_handles,
        x_key="traj_len",
        y_key="nodes",
        xlabel="Trajectory Length (m)",
        ylabel="Number of 3DSG Nodes",
        title="3DSG Nodes vs Trajectory Length",
    )

    scatter_plot(
        records,
        color_by_dataset,
        # marker_by_cameras,
        marker_by_dataset,
        legend_handles,
        x_key="memory_mb",
        y_key="nodes",
        xlabel="Memory Size (MB)",
        ylabel="Number of 3DSG Nodes",
        title="3DSG Nodes vs Terra Memory",
    )
    
    two_panel_traj_plot_vertical(
        records,
        color_by_dataset,
        # marker_by_cameras,
        marker_by_dataset,
        legend_handles,
    )
    
    scatter_plot(
        records,
        color_by_dataset,
        # marker_by_cameras,
        marker_by_dataset,
        legend_handles,
        x_key="area_m2",
        y_key="memory_mb",
        xlabel="Area (m²)",
        ylabel="Memory Size (MB)",
        title="Terra Memory vs Area",
    )

    scatter_plot(
        records,
        color_by_dataset,
        # marker_by_cameras,
        marker_by_dataset,
        legend_handles,
        x_key="area_m2",
        y_key="nodes",
        xlabel="Area (m²)",
        ylabel="Number of 3DSG Nodes",
        title="3DSG Nodes vs Area",
    )

    two_panel_area_plot_vertical(
        records,
        color_by_dataset,
        # marker_by_cameras,
        marker_by_dataset,
        legend_handles,
    )


    plt.show()


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
