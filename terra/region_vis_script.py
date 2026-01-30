from argparse import ArgumentParser
import yaml
import torch

import clip

from terra_utils import load_terra

if __name__ == '__main__':
    parser = ArgumentParser(description="Region Monitoring Test")
    parser.add_argument(
        '--params',
        type=str,
        help="/path/to/region_querying.yaml file of region monitoring tasks"
    )
    args = parser.parse_args()
    
    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()

    with open(args.params, 'rb') as f:
        region_task_params = yaml.safe_load(f)
    
    terra = load_terra(region_task_params["terra"])
    terra.visualizer.level_offset = 0.0  # Set level offset for better visualization
    
    # # Display just place nodes
    # terra.display_places(
    #     display_pc=False,
    #     plot_ids=False
    # )

    # # Display Places
    # terra.display_places(
    #     display_pc=True,
    #     plot_ids=True
    # )

    # Display just place nodes
    terra.display_places(
        display_pc=False,
        plot_ids=True,
        no_spheres=True
    )

    red_indeces = [1689, 331, 1547, 1416]
    included_nodes = terra.visualizer.get_nodes_in_rectangle_from_refs(
        terra.terra_3dsg,
        red_indeces
    )
    print(f"Red Nodes: {included_nodes}")

    terra.visualizer.display_selected_nodes(
        terra.terra_3dsg,
        included_nodes
    )

    blue_indeces = [(734, 737), (737, 1383), (1401, 1402), (2820, 746)]
    blue_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
        terra.terra_3dsg,
        blue_indeces
    )
    print(f"Blue Nodes: {blue_nodes}")
    terra.visualizer.display_selected_nodes(
        terra.terra_3dsg,
        blue_nodes
    )


    purple_indeces = [(639, 1578), (1580, 1593), (754, 2664), (644, 1549)]
    purple_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
        terra.terra_3dsg,
        purple_indeces
    )
    print(f"Purple Nodes: {purple_nodes}")
    terra.visualizer.display_selected_nodes(
        terra.terra_3dsg,
        purple_nodes
    )

