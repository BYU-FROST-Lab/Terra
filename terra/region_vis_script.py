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


    # # Green Nodes
    # green_indeces = [(272, 699), (265, 209), (204, 205), (215, 680)]
    # green_nodes = terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     green_indeces
    # )

    # green_nodes += [624, 195, 198, 10, 267]
    # print(f"Green Nodes: {green_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     green_nodes
    # )

    # # Red Nodes

    # red_nodes = [122, 123, 124, 637, 638, 511, 592, 593, 594, 1037, 1038, 639] + list(range(329, 427))
    # nodes_to_remove = [412, 411, 416, 417, 419, 330]
    # for rm_node in nodes_to_remove:
    #     if rm_node in red_nodes:
    #         red_nodes.remove(rm_node)
    # red_nodes = list(set(red_nodes))  # Get unique nodes
    # print(f"Red Nodes: {red_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     red_nodes,
    #     pc=terra.pc
    # )


    # # Blue Nodes
    # blue_nodes = [946, 943, 945, 965, 98, 99]
    # blue_nodes = blue_nodes + list(range(287, 304)) + list(range(560, 589))
    # nodes_to_remove = [301, 304]
    # for rm_node in nodes_to_remove:
    #     if rm_node in blue_nodes:
    #         blue_nodes.remove(rm_node)
    # print(f"Blue Nodes: {blue_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     blue_nodes,
    #     pc=terra.pc
    # )


    # # Purple Nodes
    # purple_indeces_1 = [(153, 536), (716, 726), (849, 864), (213, 741)]
    # purple_indeces_2 = [(652, 643), (144, 22),(153, 536), (662, 153)]
    # purple_nodes = terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     purple_indeces_1
    # )
    # purple_nodes += terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     purple_indeces_2
    # )
    # purple_nodes = list(set(purple_nodes))  # Get unique nodes
    # print(f"Purple Nodes: {purple_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     purple_nodes
    # )


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

    # purple_indeces = [(2592, 578), (579, 1448), (1460, 1471), (699, 2660)]
    # purple_nodes= terra.visualizer.get_nodes_in_polygon_from_sides(
    #     terra.terra_3dsg,
    #     purple_indeces
    # )
    # print(f"Purple Nodes: {purple_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     purple_nodes
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

    # nodes_to_add = [820, 348]
    # nodes_to_remove = [835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 353, 925, 930]
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

