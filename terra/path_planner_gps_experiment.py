import os
from argparse import ArgumentParser
import yaml
import torch
import pickle as pkl
import numpy as np
from scipy.spatial import KDTree

import clip

from terra_utils import load_terra
from utils import tensor_cosine_similarity

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


def one_prompt(clip_model, terra, path_planning_params, gps_place_nodes):
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
    planned_paths = {}
    for t_idx in range(len(path_planning_params['tasks'])):
        print("\nRunning task:", tasks[terra.num_terrain+t_idx])
        input_task_mod = input_task_clip_tensor[:(terra.num_terrain+1),:]
        input_task_mod[-1,:] = input_task_clip_tensor[terra.num_terrain+t_idx,:]
        selected_nodes = terra._select_best_place_nodes(
            input_task_mod,
            method=path_planning_params["prediction_method"],
            topk=topk
        )
        print("\nRunning task:", tasks[terra.num_terrain+t_idx])
        if 0 not in selected_nodes or len(selected_nodes[0]) == 0:
            print("NO DETECTED OBJECT!!")
            continue
        for k_idx, pid in enumerate(selected_nodes[0]):
            start_node_id = 0
            end_node_id = 0
            if k_idx == 0:
                start_node_id = start_nodes[t_idx]
                end_node_id = pid
            else:
                start_node_id = selected_nodes[0][k_idx-1]
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
        
            # Check if GT object detected
            while True:
                answer = input("Is this correct? (y/n): ").strip().lower()
                if answer == 'y':
                    print("Correct")
                    break
                elif answer == 'n':
                    print("Incorrect")
                    break
                else:
                    print("Please enter 'y' or 'n'")
            if answer == 'y':
                break
    return planned_paths

def separate_object_context(clip_model, terra, path_planning_params, gps_place_nodes, gamma_obj=0.7):
    print("Chose gamma:",gamma_obj)
    
    tasks = [task["object"] for task in path_planning_params['tasks']]
    start_nodes = [task["start_node"] for task in path_planning_params['tasks']]
    topk = path_planning_params.get("topk", 1)   
    
    # Encode prompts with CLIP
    tasks[:0] = terra.terrain_names # Add terrain to front of tasks
    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    input_task_clip_tensor.div_(input_task_clip_tensor.norm(dim=-1,keepdim=True))
    print("\nCollected destination task(s):\n", tasks)
    
    # Encode context
    contexts = [task["context"] for task in path_planning_params['tasks']]
    input_context_clip_embs = [clip_model.encode_text(clip.tokenize([context]).to(device)).float() for context in contexts]
    input_context_clip_tensor = torch.vstack(input_context_clip_embs) # (num_input_classes, 512)
    input_context_clip_tensor.div_(input_context_clip_tensor.norm(dim=-1,keepdim=True))

    # Find closest place nodes to each object
    place_nodes = [
        n for n, d in terra.terra_3dsg.nodes(data=True)
        if d["level"] == 1
    ]
    place_embeddings = torch.vstack([terra.terra_3dsg.nodes[n]["embedding"] for n in place_nodes])
    place_pos = np.array([
        terra.terra_3dsg.nodes[n]["pos"]
        for n in place_nodes
    ])
    kdt = KDTree(place_pos)
    
    pairs = {}
    for t_idx in range(len(path_planning_params['tasks'])):
        input_task_mod = input_task_clip_tensor[:(terra.num_terrain+1),:]
        input_task_mod[-1,:] = input_task_clip_tensor[terra.num_terrain+t_idx,:]
        
        input_context_mod = input_context_clip_tensor[t_idx,:]
    
        # Predict objects
        terra.predict_objects(
            input_task_mod, 
            tasks[terra.num_terrain:], 
            path_planning_params['prediction_method']
        )
        print(f"Predicted {len(terra.objects)} objects")
        if len(terra.objects) == 0:
            continue
        # terra.display_terra(display_pc=True,plot_objects_on_ground=True,color_pc_clip=False)
        
        place_scores = tensor_cosine_similarity(
            place_embeddings, 
            input_context_mod
        ) # (num_place_nodes, 1)
    
        for o_idx, o in enumerate(terra.objects):
            obb = o.get_bbox()
            o_score = o.get_top_score()
            obb_center = obb.center[:2]
            dist, pn_idx = kdt.query(obb_center)
            
            # print("No weight")
            # pn_score = place_scores[pn_idx, 0]
            
            # print("Weight @ gamma:",gamma_obj)
            pn_score = gamma_obj * o_score + (1 - gamma_obj) * place_scores[pn_idx, 0]
            
            context_idx = torch.argmax(place_scores[pn_idx, :])
            
            if t_idx not in pairs:
                pairs[t_idx] = []
            # pairs[t_idx].append({
            #     "object_idx": o_idx,
            #     "place_node_idx": pn_idx,
            #     "place_node_id": place_nodes[pn_idx],
            #     "place_node_score": pn_score.item(),
            #     "context_idx": context_idx.item()
            # })
            pairs[t_idx].append({
                "place_node_id": place_nodes[pn_idx],
                "place_node_score": pn_score.item()
            })
    
    # Select object per object task with highest place node context score
    planned_paths = {}
    for t_idx in pairs.keys():
        print("\nRunning task:", tasks[terra.num_terrain+t_idx], "with context:",contexts[t_idx])
        
        k = min(topk, len(pairs[t_idx]))
        
        # Sort list by context score
        pairs[t_idx] = sorted(
            pairs[t_idx],
            key=lambda x: x["place_node_score"],
            reverse=True
        )[:k]
        
        for k_idx, pair_dict in enumerate(pairs[t_idx]):
            pid = pair_dict["place_node_id"]
            
            start_node_id = 0
            end_node_id = 0
            if k_idx == 0:
                start_node_id = start_nodes[t_idx]
                end_node_id = pid
            else:
                start_node_id = pairs[t_idx][k_idx-1]["place_node_id"]
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
            
            # Check if GT object detected
            while True:
                answer = input("Is this correct? (y/n): ").strip().lower()
                if answer == 'y':
                    print("Correct")
                    break
                elif answer == 'n':
                    print("Incorrect")
                    break
                else:
                    print("Please enter 'y' or 'n'")
            if answer == 'y':
                break
    return planned_paths

def ensemble_of_prompts(clip_model, terra, path_planning_params, gps_place_nodes):
    task_dict = {}
    for task_idx, task in enumerate(path_planning_params['tasks']):
        task_dict[task_idx] = {}
        task_dict[task_idx]["start_node"] = task["start_node"]
        task_prompts = []
        for p in task["prompts"]:
            task_prompts.append(p)
        task_dict[task_idx]["prompts"] = task_prompts
    topk = path_planning_params.get("topk", 1)   
    
    planned_paths = {}
    for task_idx in task_dict.keys():
        tasks = task_dict[task_idx]["prompts"]
        num_prompts = len(tasks)
        
        # Encode prompts with CLIP
        tasks[:0] = terra.terrain_names # Add terrain to front of tasks
        input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
        input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
        input_task_clip_tensor.div_(input_task_clip_tensor.norm(dim=-1,keepdim=True))
        print("\nCollected destination task(s):\n", tasks)
        
        place_nodes = [
            n for n, d in terra.terra_3dsg.nodes(data=True)
            if d["level"] == 1
        ]
        pidx2pnid = {idx: id for idx, id in enumerate(terra.terra_3dsg.nodes)} 
        place_embeddings = torch.vstack([terra.terra_3dsg.nodes[n]["embedding"] for n in place_nodes])
        place_scores = tensor_cosine_similarity(
            place_embeddings, 
            input_task_clip_tensor[terra.num_terrain:,:]
        ) # (num_place_nodes, num_tasks)
        place_score_ens = place_scores.sum(dim=1) / num_prompts # num_place_nodes
        place_score_ens_mask = place_score_ens > path_planning_params["alpha"]
        place_score_ens_filt = place_score_ens[place_score_ens_mask]
        masked_indices = torch.nonzero(place_score_ens_mask)
        
        k = min(topk, torch.count_nonzero(place_score_ens_mask).item())
        
        top_vals, top_indices = torch.topk(place_score_ens_filt,k=k)
        dest_place_nodes = [pidx2pnid[masked_indices[top_idx.item()].item()] for top_idx in top_indices]
        
        for k_idx, k_pnid in enumerate(dest_place_nodes):
            start_node_id = 0
            end_node_id = 0
            if k_idx == 0:
                start_node_id = task_dict[task_idx]["start_node"]
                end_node_id = k_pnid
            else:
                start_node_id = dest_place_nodes[k_idx-1]
                end_node_id = k_pnid
            
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

            if task_idx not in planned_paths:
                planned_paths[task_idx] = []
            planned_paths[task_idx].append(chosen_place_nodes)
            terra.path_node_list = [] # reset path for next destination query 
            
            # Check if GT object detected
            while True:
                answer = input("Is this correct? (y/n): ").strip().lower()
                if answer == 'y':
                    print("Correct")
                    break
                elif answer == 'n':
                    print("Incorrect")
                    break
                else:
                    print("Please enter 'y' or 'n'")
            if answer == 'y':
                break
    return planned_paths


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
    
    print("Running", path_planning_params["unique_object_approach"], "approach")
    if path_planning_params["unique_object_approach"] == "1-short" or path_planning_params["unique_object_approach"] == "1-long":
        planned_paths = one_prompt(
            clip_model, 
            terra,
            path_planning_params,
            gps_place_nodes
        ) # {task_idx: [[...], ...]}
        task_names_dict = [task["query_destination"] for task in path_planning_params['tasks']]
    elif path_planning_params["unique_object_approach"] == "object+context":
        planned_paths = separate_object_context(
            clip_model, 
            terra,
            path_planning_params,
            gps_place_nodes,
            path_planning_params["gamma_obj"]
        )
        task_names_dict = [task["object"]+": "+task["context"] for task in path_planning_params['tasks']]
    # elif path_planning_params["unique_object_approach"] == "ensemble":
    #     planned_paths = ensemble_of_prompts(
    #         clip_model,
    #         terra,
    #         path_planning_params,
    #         gps_place_nodes
    #     )
    
    gps_dict = {}
    enu_dict = {}
    for task_idx in planned_paths.keys():
        gps_dict[task_idx] = {}
        gps_dict[task_idx]["task"] = task_names_dict[task_idx]
        enu_dict[task_idx] = {}
        enu_dict[task_idx]["task"] = task_names_dict[task_idx]
        print("\n\nTask:",gps_dict[task_idx]["task"])
        
        correct_k = len(planned_paths[task_idx])
        for k in range(correct_k):
            print(f"\nPlace node path for k={k}")
            place_node_ids = planned_paths[task_idx][k]   
            
            gps_dict[task_idx][k] = []
            enu_dict[task_idx][k] = []
            for pid in place_node_ids:
                gps = gps_place_nodes[pid][:2] # (lat, lon)
                print(gps)
                enu = enu_place_nodes[pid][:2] # (x_east,y_north)
                gps_dict[task_idx][k].append(gps)
                enu_dict[task_idx][k].append(enu)
            # print("GPS coords:\n",gps_dict[task_idx][k])
            
    # Save GPS coordinates of the predicted path to the destination for each task
    output_dir = os.path.dirname(path_planning_params["place_nodes_gps"])
    orig_name = os.path.basename(path_planning_params["place_nodes_gps"])
    new_gps_name = "path_gps" + orig_name[8:]
    new_enu_name = "path_enu" + orig_name[8:]
    
    with open(os.path.join(output_dir,new_gps_name), 'wb') as f:
        pkl.dump(gps_dict, f) # {0: {"task": "...", 0: [[40,-111.3], ...], ...}, 1: ...}
    with open(os.path.join(output_dir,new_enu_name), 'wb') as f:
        pkl.dump(enu_dict, f) # {0: {"task": "...", 0: [[1001.1,-1101.3], ...], ...}, 1: ...}
        
    print("Saved GPS/ENU coordinates of path planner")