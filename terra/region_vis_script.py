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

    # # Display just place nodes
    # terra.display_places(
    #     display_pc=False,
    #     plot_ids=True,
    #     no_spheres=True
    # )

    # red_indeces = [1689, 331, 1547, 1416]
    # included_nodes = terra.visualizer.get_nodes_in_rectangle_from_refs(
    #     terra.terra_3dsg,
    #     red_indeces
    # )
    # print(f"Red Nodes: {included_nodes}")

    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     included_nodes
    # )

    # blue_indeces = [(734, 737), (737, 1383), (1401, 1402), (2820, 746)]
    # blue_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     blue_indeces
    # )
    # print(f"Blue Nodes: {blue_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     blue_nodes
    # )


    red_indeces = [(28, 35), (940, 953), (964, 1043), (27, 906)]
    red_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
        terra.terra_3dsg,
        red_indeces
    )

    nodes_to_remove = [27, 26, 25, 24, 23]
    nodes_to_add = [28, 35, 940, 953, 964, 1043, 27, 906]
    red_nodes = red_nodes + nodes_to_add
    for rm_node in nodes_to_remove:
        if rm_node in red_nodes:
            red_nodes.remove(rm_node)
    red_nodes = list(set(red_nodes))  # Get unique nodes
    print(f"Red Nodes: {red_nodes}")
    terra.visualizer.display_selected_nodes(
        terra.terra_3dsg,
        red_nodes
    )

    # big_box_corners = [50, 765]
    # small_box_corners = [765, 997]
    # big_box_nodes = terra.visualizer.get_nodes_in_diagonal_rectangle(
    #     terra.terra_3dsg,
    #     big_box_corners
    # )
    # small_box_nodes = terra.visualizer.get_nodes_in_diagonal_rectangle(
    #     terra.terra_3dsg,
    #     small_box_corners
    # )

    # nodes_to_add = [1001, 743]
    # nodes_to_remove = [782, 783, 784, 785, 786, 787, 788, 789, 972, 975, 690, 696]
    # combined_nodes = big_box_nodes + small_box_nodes + nodes_to_add
    # for rm_node in nodes_to_remove:
    #     if rm_node in combined_nodes:
    #         combined_nodes.remove(rm_node)

    # #Get unique nodes
    # combined_nodes = list(set(combined_nodes))
    # print(f"Combined Nodes: {combined_nodes}")

    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     combined_nodes
    # )

