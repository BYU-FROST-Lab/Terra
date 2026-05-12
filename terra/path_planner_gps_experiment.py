from argparse import ArgumentParser
import yaml
import torch
import pickle as pkl

import clip

from terra_utils import load_terra

'''
Runs path planner offline and saves GPS coordinates for place nodes

Input:
    - path_planning.yaml file of path planning tasks, which includes:
        - starting GPS location
        - destination query (e.g. "a park with a playground")

Output:
    - GPS coordinates of the predicted path to the destination

Example usage:
    python3 path_planner_gps_experiment.py --params /path/to/path_planning.yaml
'''

if __name__ == '__main__':
    parser = ArgumentParser(description="Path Planning Test")
    parser.add_argument(
        '--params',
        type=str,
        help="/path/to/path_planning.yaml file of path planning tasks"
    )
    args = parser.parse_args()

    with open(args.params, 'rb') as f:
        path_planning_params = yaml.safe_load(f)
    
    # Load GPS Place Node dictionary
    with open(path_planning_params["place_nodes_gps"], 'rb') as f:
        gps_place_nodes = pkl.load(f) # {place_node_id: gps_coordinates, ...}
    
    # Load ENU Place Node dictionary
    with open(path_planning_params["place_nodes_enu"], 'rb') as f:
        enu_place_nodes = pkl.load(f) # {place_node_id: enu_coordinates, ...}
    
    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()
    
    terra = load_terra(path_planning_params["terra"])
    terra.alpha = path_planning_params["alpha"]

    tasks = [task["query_destination"] for task in path_planning_params['tasks']]
    start_nodes = [task["start_node"] for task in path_planning_params['tasks']]
    topk = path_planning_params.get("topk", 1)   
    
    # Encode prompts with CLIP
    tasks[:0] = terra.terrain_names # Add terrain to front of tasks
    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    input_task_clip_tensor.div_(input_task_clip_tensor.norm(dim=-1,keepdim=True))
    print("\nCollected destination task(s):\n", tasks)
    
    # Select n-best place nodes
    selected_nodes = terra._select_best_place_nodes(
        input_task_clip_tensor,
        method=path_planning_params["prediction_method"],
        topk=topk
    )
    planned_paths = {}
    for t_idx in range(len(path_planning_params['tasks'])):
        for k_idx, pid in enumerate(selected_nodes[t_idx]):
            start_node_id = 0
            end_node_id = 0
            if k_idx == 0:
                start_node_id = start_nodes[t_idx]
                end_node_id = pid
            else:
                start_node_id = selected_nodes[t_idx][k_idx-1]
                end_node_id = pid
            
            try:
                print(f"Start place node {start_node_id} with GPS {gps_place_nodes[start_node_id]}")
                print(f"Destination place node {end_node_id} with GPS {gps_place_nodes[end_node_id]}")                       
            except KeyError as e:
                print(f"Error: Place node {e} not found in GPS dictionary")

            terra._plan_path_astar(
                start_node=start_node_id,
                dest_node=end_node_id,
                terrain_preferences=path_planning_params["terrain_preferences"]
            )
            terra.display_path() # black = start, red = destination
                        
            chosen_place_nodes = terra.path_node_list # list of place node ids

            if t_idx not in planned_paths:
                planned_paths[t_idx] = []
            planned_paths[t_idx].append(chosen_place_nodes)
            terra.path_node_list = [] # reset path for next destination query
        
        # # Display top-k paths for this task
        # terra.visualizer.display_multi_paths(
        #     terra.terra_3dsg,
        #     planned_paths[t_idx], 
        #     terra.pc
        # )
        
    # TODO: Save GPS coordinates of the predicted path to the destination for each task

    print("Finished")