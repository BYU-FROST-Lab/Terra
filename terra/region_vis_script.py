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
        plot_ids=True
    )

    # Display just place nodes to read ids
    terra.display_places(
        display_pc=False,
        plot_ids=True,
        no_spheres=True
    )

    # red_indeces = [1689, 331, 1547, 1416]
    # included_nodes = terra.visualizer.get_nodes_in_rectangle_from_refs(
    #     terra.terra_3dsg,
    #     red_indeces
    # )
    # print(f"Red Nodes: {included_nodes}")

    # included_nodes = [119, 120, 121, 122, 123, 124, 125, 129, 130, 132, 131, 133, 134, 137, 138, 140, 141, 34, 35, 32, 33, 22, 23, 21, 24, 16, 17, 19, 14, 15, 13, 20, 18, 439, 443, 479, 489, 495, 505, 506, 512, 514, 526, 515, 508, 539, 540, 545, 563, 566, 565, 566, 578, 590, 591, 592, 619, 627, 609, 604, 637, 610, 592, 631, 638, 640, 644, 661, 672, 571, 581, 587, 600, 456, 465, 483, 499, 527, 523, 528, 529, 536, 547, 541, 516, 509, 497, 203, 204, 172, 173, 184, 186, 187]
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     included_nodes,
    #     pc=terra.pc
    # )

    # purple_indeces = [(287, 136), (245, 289), (315, 289), (314, 352)]
    # purple_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     purple_indeces
    # )
    # print(f"Purple Nodes: {purple_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     purple_nodes
    # )

    # green_indeces = [(416, 139), (146, 483), (125, 607), (679, 125)]
    # green_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     green_indeces
    # )
    # print(f"Green Nodes: {green_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     green_nodes
    # )


    # red_indeces = [(28, 35), (940, 953), (964, 1043), (27, 906)]
    # red_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     red_indeces
    # )

    # nodes_to_remove = [27, 26, 25, 24, 23]
    # nodes_to_add = [28, 35, 940, 953, 964, 1043, 27, 906]
    # red_nodes = red_nodes + nodes_to_add
    # for rm_node in nodes_to_remove:
    #     if rm_node in red_nodes:
    #         red_nodes.remove(rm_node)
    # red_nodes = list(set(red_nodes))  # Get unique nodes
    # print(f"Red Nodes: {red_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     red_nodes
    # )

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

