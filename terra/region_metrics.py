from argparse import ArgumentParser
import time
import yaml
import torch
import numpy as np
import clip

from terra_utils import load_terra


def compute_region_metrics(pred_places, gt_places):
    tps, fps, fns = 0,0,0
    for query_idx in gt_places.keys():
        if query_idx in pred_places:
            pred_set = set(pred_places[query_idx])
        else:
            pred_set = set()
        gt_set = set(gt_places[query_idx])

        tp, fp, fn, tn = compute_confusion_matrix(pred_set, gt_set)
        tps += tp
        fps += fp
        fns += fn
    
    prec, rec, f1 = compute_precision_recall_f1(tps, fps, fns)
    return prec, rec, f1

def compute_precision_recall_f1(true_positives, false_positives, false_negatives):
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def compute_confusion_matrix(pred_places_set, gt_places_set):
    true_positives = len(gt_places_set & pred_places_set)
    false_positives = len(pred_places_set - gt_places_set)
    false_negatives = len(gt_places_set - pred_places_set)
    true_negatives = 0  # Not computable without total population size
    
    return true_positives, false_positives, false_negatives, true_negatives

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
    terra.alpha = region_task_params["alpha"]
    region_tasks = region_task_params["region_tasks"]

    # place_nodes = [n for n, d in terra.terra_3dsg.nodes(data=True) if d["level"] == 1]
    # print("Number of nodes in the dataset:", len(place_nodes))

    # print("Number of ground truth place nodes for each query:")
    # for idx, task in enumerate(region_tasks):
    #     gt_place_nodes = task["place_nodes"]
    #     print(f"Query {idx} - {task['task']}: {len(gt_place_nodes)} place nodes")

    # Encode prompts with CLIP
    tasks = []
    place_nodes_dict = {}
    for task_index, task in enumerate(region_tasks):
        tasks.append(task["task"])
        place_nodes = task["place_nodes"]

        #Clean place nodes to remove potential region nodes
        for pn in place_nodes:
            level = terra.terra_3dsg.nodes[pn]["level"]
            if level != 1:
                place_nodes.remove(pn)
        
        place_nodes_dict[task_index] = place_nodes


    input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in tasks]
    input_task_clip_tensor = torch.vstack(input_task_clip_embs) # (num_input_classes, 512)
    input_task_clip_tensor.div_(input_task_clip_tensor.norm(dim=-1,keepdim=True))
    print("\nCollected region tasks:", tasks)

    alpha_values = np.linspace(0.2, 0.35, 16)
    k_values = np.linspace(1, 10, 10)
    best_alpha = None
    best_k = None
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    # Find the best alpha and k based on evaluation metrics
    for alpha in alpha_values:
        terra.alpha = alpha
        for k in k_values:
            print(f"\nPredicting regions with alpha: {alpha}, k: {k}")
            terra.reset_region_tasks()
            # Prediction regions given prompts
            terra.predict_regions(
                input_task_clip_tensor, 
                tasks, 
                region_task_params["prediction_method"], 
                K = int(k)
            )

            pred_places = terra.task_relevant_place_nodes

            precision, recall, f1 = compute_region_metrics(pred_places, place_nodes_dict)
            print(f"Alpha: {alpha}, K: {k} => Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Update best parameters
            if f1 > best_f1:
                best_f1 = f1
                best_alpha = alpha
                best_k = k
                best_precision = precision
                best_recall = recall

    print(f"\nBest parameters found - Alpha: {best_alpha}, K: {best_k} with Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    
    #Predict and Display Best Results
    terra.reset_region_tasks()
    terra.alpha = best_alpha

    start_time = time.time()
    terra.predict_regions(
        input_task_clip_tensor, 
        tasks, 
        region_task_params["prediction_method"], 
        K = int(best_k)
    )
    end_time = time.time()
    print(f"\nRuntime (ms): {(end_time - start_time)*1000:.2f}")

    # Display ground truth place nodes for each query
    # print("Ground Truth Place Nodes:")

    # for task_idx, place_nodes in place_nodes_dict.items():
    #     print(f"Task: {tasks[task_idx]}, Place Nodes: {place_nodes}")

    #     terra.visualizer.display_selected_nodes(
    #         terra.terra_3dsg,
    #         place_nodes,
    #         pc=terra.pc
    #     )

    for task_idx in range(len(region_tasks)):
        terra.display_task_relevant_places(task_idx, heatmap_mode=True)
    terra.display_task_relevant_places()

