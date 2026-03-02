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
    
    # Display just place nodes
    terra.display_places(
        display_pc=False,
        plot_ids=False
    )

    # Display Place nodes with pc
    terra.display_places(
        display_pc=True,
        plot_ids=False
    )

    # Display just place nodes to read ids
    terra.display_places(
        display_pc=False,
        plot_ids=True,
        no_spheres=True
    )


    # #### Two point on each side ####
    # blue_indeces = [(948, 792), (714, 746), (746, 1247), (1267, 2823)]
    # blue_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     blue_indeces
    # )
    # print(f"Blue Nodes: {blue_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     blue_nodes
    # )


    # #### Two points on diagonal corners ####
    # big_box_corners = [25, 949]
    # small_box_corners = [949, 353]
    # big_box_nodes = terra.visualizer.get_nodes_in_diagonal_rectangle(
    #     terra.terra_3dsg,
    #     big_box_corners
    # )
    # small_box_nodes = terra.visualizer.get_nodes_in_diagonal_rectangle(
    #     terra.terra_3dsg,
    #     small_box_corners
    # )


    # combined_nodes = big_box_nodes + small_box_nodes 
    # #Get unique nodes
    # combined_nodes = list(set(combined_nodes))
    # print(f"Combined Nodes: {combined_nodes}")

    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     combined_nodes
    # )

