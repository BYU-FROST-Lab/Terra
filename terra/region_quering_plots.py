from argparse import ArgumentParser
import time
import yaml
import torch
import numpy as np
import clip
import matplotlib.pyplot as plt

from terra_utils import load_terra


def compute_region_metrics(pred_places, gt_places):
    tps, fps, fns = 0,0,0
    task_metrics = {}
    for query_idx in gt_places.keys():
        if query_idx in pred_places:
            pred_set = set(pred_places[query_idx])
        else:
            pred_set = set()
        gt_set = set(gt_places[query_idx])

        tp, fp, fn, tn = compute_confusion_matrix(pred_set, gt_set)
        task_metrics[query_idx] = compute_precision_recall_f1(tp, fp, fn)
        tps += tp
        fps += fp
        fns += fn
    
    overall_prec, overall_rec, overall_f1 = compute_precision_recall_f1(tps, fps, fns)
    return overall_prec, overall_rec, overall_f1, task_metrics

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
        '--params_1cam',
        type=str,
        help="/path/to/1cam_region_querying.yaml file of region monitoring tasks"
    )
    parser.add_argument(
        '--params_3cam',
        type=str,
        help="/path/to/3cam_region_querying.yaml file of region monitoring tasks"
    )
    args = parser.parse_args()
    
    with open(args.params_1cam, 'rb') as f:
        region_task_params_1cam = yaml.safe_load(f)
    
    with open(args.params_3cam, 'rb') as f:
        region_task_params_3cam = yaml.safe_load(f)

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    logit_scale = clip_model.logit_scale.exp()

    class RegionQueryingConfig:
        def __init__(self, name, terra_path, region_tasks, prediction_method):
            self.name = name
            self.terra = load_terra(terra_path)
            self.region_tasks = region_tasks
            self.prediction_method = prediction_method
            self.f1_scores = []
            self.alphas = []
            self.ks = []
            self.best_f1 = 0.0
            self.best_alpha = 0.0
            self.best_k = 0
            self.tasks = []
            self.place_nodes_dict = {}
            self.input_task_clip_embs = None
            self.input_task_clip_tensor = None
            self.task_metrics = []


    agg_3cam_config = RegionQueryingConfig(
        name = "Agglomerative 3 Cam",
        terra_path=region_task_params_3cam["terra"],
        region_tasks=region_task_params_3cam["region_tasks"],
        prediction_method=region_task_params_3cam["prediction_method"]
    )

    agg_1cam_config = RegionQueryingConfig(
        name = "Agglomerative 1 Cam",
        terra_path=region_task_params_1cam["terra"],
        region_tasks=region_task_params_1cam["region_tasks"],
        prediction_method=region_task_params_1cam["prediction_method"]
    )

    spec_3cam_terra_path = region_task_params_3cam["terra"].replace("agglomerative", "spectral")
    spec_3cam_config = RegionQueryingConfig(
        name = "Spectral 3 Cam",
        terra_path=spec_3cam_terra_path,
        region_tasks=region_task_params_3cam["region_tasks"],
        prediction_method=region_task_params_3cam["prediction_method"]
    )

    spec_1cam_terra_path = region_task_params_1cam["terra"].replace("agglomerative", "spectral")
    spec_1cam_config = RegionQueryingConfig(
        name = "Spectral 1 Cam",
        terra_path=spec_1cam_terra_path,
        region_tasks=region_task_params_1cam["region_tasks"],
        prediction_method=region_task_params_1cam["prediction_method"]
    )

    configs = [
        agg_3cam_config,
        agg_1cam_config,
        spec_3cam_config,
        spec_1cam_config
    ]

    for config in configs:
        for task_index, task in enumerate(config.region_tasks):
            config.tasks.append(task["task"])
            place_nodes = task["place_nodes"]

            #Clean place nodes to remove potential region nodes
            for pn in place_nodes:
                level = config.terra.terra_3dsg.nodes[pn]["level"]
                if level != 1:
                    place_nodes.remove(pn)
            
            config.place_nodes_dict[task_index] = place_nodes

        config.input_task_clip_embs = [clip_model.encode_text(clip.tokenize([tsk]).to(device)).float() for tsk in config.tasks]
        config.input_task_clip_tensor = torch.vstack(config.input_task_clip_embs) # (num_input_classes, 512)
        print(f"\nCollected region tasks for {config.name}:", config.tasks)

    

    alpha_values = np.linspace(0.2, 0.35, 16)
    k_values = np.linspace(1, 11, 11)


    # Find the best alpha and k based on evaluation metrics
    for config in configs:
        print(f"\nEvaluating configuration: {config.name}")
        for alpha in alpha_values:
            config.terra.alpha = alpha
            for k in k_values:
                print(f"Predicting regions with alpha: {alpha}, k: {k}")
                config.terra.reset_region_tasks()
                # Prediction regions given prompts
                config.terra.predict_regions(
                    config.input_task_clip_tensor, 
                    config.tasks, 
                    config.prediction_method, 
                    K = int(k)
                )

                pred_places = config.terra.task_relevant_place_nodes

                precision, recall, f1, task_metrics = compute_region_metrics(pred_places, config.place_nodes_dict)
                config.f1_scores.append(f1)
                config.alphas.append(alpha)
                config.ks.append(k)
                config.task_metrics.append(task_metrics)
                print(f"Alpha: {alpha}, K: {k} => Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
                # Update best parameters
                if f1 > config.best_f1:
                    config.best_f1 = f1
                    config.best_alpha = alpha
                    config.best_k = k
                    config.best_precision = precision
                    config.best_recall = recall

        print(f"\nBest parameters found - Alpha: {config.best_alpha}, K: {config.best_k} with Precision: {config.best_precision:.4f}, Recall: {config.best_recall:.4f}, F1: {config.best_f1:.4f}")

    # for config in configs:
    #     print(f"\nFinal Best parameters for {config.name} - Alpha: {config.best_alpha}, K: {config.best_k} with Precision: {config.best_precision:.4f}, Recall: {config.best_recall:.4f}, F1: {config.best_f1:.4f}")
    #     print("Best per Task Metrics:")
    #     for task_idx, task in enumerate(config.region_tasks):
    #         task_prec, task_rec, task_f1 = config.task_metrics[config.alphas.index(config.best_alpha) + config.ks.index(config.best_k)*len(alpha_values)][task_idx]
    #         print(f"  Task: {task['task']}")
    #         print(f"    Precision: {task_prec:.4f}, Recall: {task_rec:.4f}, F1: {task_f1:.4f}")

    #Plot grid of 4 plots of F1 vs Alpha with best K for each configuration
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for i, config in enumerate(configs):
        ax = axs[i//2, i%2]
        best_ks = []
        best_f1s = []
    
        f1s_for_alpha = [f1 for a, k, f1 in zip(config.alphas, config.ks, config.f1_scores) if k == config.best_k]

        ax.plot(alpha_values, f1s_for_alpha, marker='o')
        ax.set_title(f'F1 vs Alpha - {config.name}(K={int(config.best_k)})')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('F1 Score')
        ax.grid(True)
    
    plt.show()

    #Plot grid of 4 plots of F1 vs K with best Alpha for each configuration
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for i, config in enumerate(configs):
        ax = axs[i//2, i%2]
        best_ks = []
        best_f1s = []
    
        f1s_for_k = [f1 for a, k, f1 in zip(config.alphas, config.ks, config.f1_scores) if a == config.best_alpha]

        ax.plot(k_values, f1s_for_k, marker='o')
        ax.set_title(f'F1 vs K - {config.name}(Alpha={config.best_alpha})')
        ax.set_xlabel('K')
        ax.set_ylabel('F1 Score')
        ax.grid(True)

    plt.show()

