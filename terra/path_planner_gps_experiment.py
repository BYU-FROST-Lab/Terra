import os
from argparse import ArgumentParser
import yaml
import torch
import re
import pickle as pkl
import numpy as np
from scipy.spatial import KDTree
from math import radians, sin, cos, sqrt, atan2
import cv2
import matplotlib.pyplot as plt

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

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute distance in meters between two GPS coordinates.
    """
    R = 6371000  # Earth radius in meters

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def place_node_from_gps(gps_coord, gps_place_node_dict):
    target_lat, target_lon = gps_coord
    
    closest_id = None
    min_dist = float("inf")
    for pid, (lat,lon) in gps_place_node_dict.items():
        dist = haversine_distance(target_lat, target_lon, lat, lon)
        if dist < min_dist:
            min_dist = dist
            closest_id = pid
    return closest_id

def one_prompt(clip_model, terra, path_planning_params):
    tasks = [task["query_destination"] for task in path_planning_params['tasks']]
    topk = path_planning_params.get("topk", 1)   
    
    # Encode prompts with CLIP
    tasks[:0] = terra.terrain_names # Add terrain to front of tasks
    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    input_task_clip_tensor.div_(input_task_clip_tensor.norm(dim=-1,keepdim=True))
    print("\nCollected destination task(s):\n", tasks)
    
    # Select n-best place nodes
    dest_nodes = {}
    for t_idx in range(len(path_planning_params['tasks'])):
        input_task_mod = input_task_clip_tensor[:(terra.num_terrain+1),:]
        input_task_mod[-1,:] = input_task_clip_tensor[terra.num_terrain+t_idx,:]
        selected_nodes = terra._select_best_place_nodes(
            input_task_mod,
            method=path_planning_params["prediction_method"],
            topk=topk
        )
        if 0 not in selected_nodes or len(selected_nodes[0]) == 0:
            print("NO DETECTED OBJECT!!")
            continue
        dest_nodes[t_idx] = selected_nodes[0]
    return dest_nodes

def separate_object_context(clip_model, terra, path_planning_params, gamma_obj=0.7):
    print("gamma_obj:",gamma_obj)
    
    tasks = [task["object"] for task in path_planning_params['tasks']]
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
        # terra.display_terra(
        #     display_pc=True,
        #     plot_objects_on_ground=True,
        #     color_pc_clip=False
        # )
        
        place_scores = tensor_cosine_similarity(
            place_embeddings, 
            input_context_mod
        ) # (num_place_nodes, 1)
    
        for o_idx, o in enumerate(terra.objects):
            obb = o.get_bbox()
            o_score = o.get_top_score()
            obb_center = obb.center[:2]
            dist, pn_idx = kdt.query(obb_center)
            
            pn_score = gamma_obj * o_score + (1 - gamma_obj) * place_scores[pn_idx, 0]
                        
            if t_idx not in pairs:
                pairs[t_idx] = []
            pairs[t_idx].append({
                "place_node_id": place_nodes[pn_idx],
                "place_node_score": pn_score.item()
            })
    
    # Select object per object task with highest place node context score
    dest_nodes = {}
    for t_idx in pairs.keys():        
        k = min(topk, len(pairs[t_idx]))
        
        # Sort list by context score
        pairs[t_idx] = sorted(
            pairs[t_idx],
            key=lambda x: x["place_node_score"],
            reverse=True
        )[:k]
        
        dest_nodes[t_idx] = [pair_dict["place_node_id"] for pair_dict in pairs[t_idx]]
        
    return dest_nodes

def display_destination_place_node_imgs(dest_node_id, terra, data_folder, num_cams=3):
    cam_image_folders = [os.path.join(data_folder, f"camera{i}_images") for i in range(1,num_cams+1)]
    
    # Images that see this node
    img_indices = terra.nodeid_2_img_idx[dest_node_id]

    shown = 0
    img_buffer = []
    
    for img_idx in img_indices:
        if shown >= 27:
            break
        img_path = terra.img_names[img_idx]
        cam_id, timestamp = parse_camera_and_timestamp(img_path)

        img_full_path = os.path.join(
            cam_image_folders[cam_id],
            os.path.basename(img_path)
        )
        if not os.path.exists(img_full_path):
            continue

        dist_img = cv2.imread(img_full_path)
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
        
        title = f"{os.path.basename(img_path)} | place {dest_node_id}"
        img_buffer.append((dist_img, title))

        shown += 1
        
        if len(img_buffer) == 9:
            show_image_batch(img_buffer)
            img_buffer.clear()
    
    if img_buffer:
        show_image_batch(img_buffer)
    plt.show()

def show_image_batch(img_buffer):
    """
    Display up to 9 images in a 3x3 subplot figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 15))
    axes = axes.flatten()
    for ax, (img, title) in zip(axes, img_buffer):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    # Hide unused subplots
    for ax in axes[len(img_buffer):]:
        ax.axis("off")
    # fig.suptitle(
    #     title,
    #     fontsize=14
    # )
    fig.tight_layout(rect=[0, 0, 1, 1])

def parse_camera_and_timestamp(img_path):
    img_name = os.path.basename(img_path)
    m = re.match(r"cam(\d+)_img_([0-9]+\.[0-9]+)", img_name)
    if m is None:
        raise ValueError(f"Cannot parse {img_name}")
    cam_id = int(m.group(1)) - 1  # 0-indexed
    timestamp = float(m.group(2))
    return cam_id, timestamp


def plan_paths(terra, terrain_preferences, start_node, dest_nodes, data_folder=None):
    planned_paths = []
    for k_idx, pid in enumerate(dest_nodes):
        start_node_id = 0
        end_node_id = 0
        if k_idx == 0:
            start_node_id = start_node
            end_node_id = pid
        else:
            start_node_id = dest_nodes[k_idx-1]
            end_node_id = pid
        
        try:
            print(f"Start place node {start_node_id}")
            print(f"Destination place node {end_node_id}")                       
        except KeyError as e:
            print(f"Error: Place node {e} not found in GPS dictionary")

        terra._plan_path_astar(
            start_node=start_node_id,
            dest_node=end_node_id,
            terrain_preferences=terrain_preferences
        )
        terra.display_path() # black = start, red = destination
        display_destination_place_node_imgs(end_node_id, terra, data_folder)
        
        chosen_place_nodes = terra.path_node_list # list of place node ids

        planned_paths.append(chosen_place_nodes)
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
    
    for start_node in path_planning_params["start_nodes"]:
        if isinstance(start_node, list):
            selected_place_node = place_node_from_gps(start_node, gps_place_nodes)
            start_node = selected_place_node
        print("\n\nStart node:",start_node)
    
        ## Identify top-k destination nodes per task
        if path_planning_params["unique_object_approach"] == "1_short" or path_planning_params["unique_object_approach"] == "1_long":
            dest_nodes = one_prompt(
                clip_model, 
                terra,
                path_planning_params
            ) # {task_idx: [node_id_0, ...]}
            task_names_dict = [task["query_destination"] for task in path_planning_params['tasks']]
        elif path_planning_params["unique_object_approach"] == "object_context":
            dest_nodes = separate_object_context(
                clip_model, 
                terra,
                path_planning_params,
                path_planning_params["gamma_obj"]
            ) # {task_idx: [node_id_0, ...]}
            task_names_dict = [task["object"]+": "+task["context"] for task in path_planning_params['tasks']]
        else:
            print("Unrecognized object approach")
            exit()
            
        ## Plan paths
        planned_paths = {}
        for t_idx in dest_nodes.keys():
            print("\nRunning task:", task_names_dict[t_idx])
            planned_paths[t_idx] = plan_paths(
                terra, 
                path_planning_params["terrain_preferences"], 
                start_node, 
                dest_nodes[t_idx],
                path_planning_params["data_folder"]
            )

        ## Convert place node ids to GPS/ENU coords
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
                
        ## Save GPS coordinates of the predicted path to the destination for each task
        output_dir = os.path.dirname(path_planning_params["place_nodes_gps"])
        orig_name = os.path.basename(path_planning_params["place_nodes_gps"])
        new_gps_name = "path_gps_" + path_planning_params["unique_object_approach"] + "_" + str(start_node) + "initnode" + orig_name[8:]
        new_enu_name = "path_enu_" + path_planning_params["unique_object_approach"] + "_" + str(start_node) + "initnode" + orig_name[8:]
        
        with open(os.path.join(output_dir,new_gps_name), 'wb') as f:
            pkl.dump(gps_dict, f) # {0: {"task": "...", 0: [[40,-111.3], ...], ...}, 1: ...}
        with open(os.path.join(output_dir,new_enu_name), 'wb') as f:
            pkl.dump(enu_dict, f) # {0: {"task": "...", 0: [[1001.1,-1101.3], ...], ...}, 1: ...}
        
    print("\nSaved GPS/ENU coordinates of path planner\n")