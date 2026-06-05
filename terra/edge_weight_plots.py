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
        task_prec, task_rec, task_f1 = compute_precision_recall_f1(tp, fp, fn)
        task_metrics[query_idx] = (task_prec, task_rec, task_f1, tp, fp, fn)
        tps += tp
        fps += fp
        fns += fn

    macro_prec = np.mean([m[0] for m in task_metrics.values()])
    macro_rec = np.mean([m[1] for m in task_metrics.values()])
    macro_f1 = np.mean([m[2] for m in task_metrics.values()])
    macro_f1_std = np.std([m[2] for m in task_metrics.values()])
    macro_prec_std = np.std([m[0] for m in task_metrics.values()])
    macro_rec_std = np.std([m[1] for m in task_metrics.values()])
    
    overall_prec, overall_rec, overall_f1 = compute_precision_recall_f1(tps, fps, fns)
    return overall_prec, overall_rec, overall_f1, task_metrics, macro_prec, macro_rec, macro_f1, macro_prec_std, macro_rec_std, macro_f1_std

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

def print_number_of_place_and_region_nodes(config):
    place_nodes = [n for n, d in config.terra.terra_3dsg.nodes(data=True) if d["level"] == 1]
    region_nodes = [n for n, d in config.terra.terra_3dsg.nodes(data=True) if d["level"] == 2]
    super_region_nodes = [n for n, d in config.terra.terra_3dsg.nodes(data=True) if d["level"] == 3]
    super_super_region_nodes = [n for n, d in config.terra.terra_3dsg.nodes(data=True) if d["level"] == 4]
    overall_region_nodes = [n for n, d in config.terra.terra_3dsg.nodes(data=True) if d["level"] > 1]
    print(f"Number of place nodes: {len(place_nodes)}, region nodes: {len(region_nodes)}, super region nodes: {len(super_region_nodes)}, super super region nodes: {len(super_super_region_nodes)}, overall region nodes: {len(overall_region_nodes)}")

    return

def clean_name(name):
    if "utah_lake_park_p1" in name:
        name = "Marina Part 1"
    elif "utah_lake_park_p2" in name:
        name = "Marina Part 2"
    elif "rock_canyon_campground" in name:
        name = "Rock Canyon Campground"
    elif "nunns_park" in name:
        name = "Nunns Park"
    elif "provo_river_p1" in name:
        name = "River Park"    
    return name

def plot_k_graphs(configs, k_values, method="micro"):
    #Plot grid of 2 by num_configs/2 plots of F1 vs K with best Alpha for each configuration
    # fig, axs = plt.subplots(2, 1, figsize=(8, 12))
    fig, axs = plt.subplots(1, 1, figsize=(8, 10))
    # fig.suptitle(f'F1-{method.capitalize()} vs K', fontsize=18)

    sums_f1s_for_k = [0.0 for _ in k_values]
    spectral_sums_f1s_for_k = [0.0 for _ in k_values]
    agglomerative_sums_f1s_for_k = [0.0 for _ in k_values]

    print(f"\n\nPrec, Rec, F1 for each config- {method}:")
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for iconfig, config in enumerate(configs):
        #plot a different color per config
        color = colors[iconfig % len(colors)]
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

        # Plot per task
        for task_idx, task in enumerate(config.region_tasks):
            task_f1s_for_k = [task_metrics[task_idx][2] for a, k, task_metrics in zip(config.alphas, config.ks, config.task_metrics) if a == config.best_alpha]
            marker = markers[task_idx % len(markers)]
            # if method == "micro":
            #     task_f1s_for_k = [f1 for a, k, f1 in zip(config.alphas, config.ks, config.micro_f1_scores) if a == config.best_alpha]
            # else: # method == "macro"
                
            #     task_f1s_for_k = [f1 for a, k, f1 in zip(config.alphas, config.ks, config.macro_f1s) if a == config.macro_best_alpha]

            

        # if config.prediction_method == "aib":
        #     continue  # Skip AIB for K plots since it doesn't vary with alpha
        

    
        # if method == "micro":
        #     f1s_for_k = [f1 for a, k, f1 in zip(config.alphas, config.ks, config.micro_f1_scores) if a == config.best_alpha]
        #     precisions_for_k = [prec for a, k, prec in zip(config.alphas, config.ks, config.micro_precisions) if a == config.best_alpha]
        #     recalls_for_k = [rec for a, k, rec in zip(config.alphas, config.ks, config.micro_recalls) if a == config.best_alpha]
        # elif method == "macro":
        #     f1s_for_k = [f1 for a, k, f1 in zip(config.alphas, config.ks, config.macro_f1s) if a == config.best_alpha]
        #     precisions_for_k = [prec for a, k, prec in zip(config.alphas, config.ks, config.macro_precisions) if a == config.best_alpha]
        #     recalls_for_k = [rec for a, k, rec in zip(config.alphas, config.ks, config.macro_recalls) if a == config.best_alpha]
        #     # f1_stds_for_k = [std for a, k, std in zip(config.alphas, config.ks, config.macro_f1_stds) if a == config.best_alpha]

        # for k, f1 in zip(config.ks, config.f1_scores):
        #     idx = np.where(k_values == k)[0][0]
        #     sums_f1s_for_k[idx] += f1
        #     if "spectral" in config.name.lower():
        #         spectral_sums_f1s_for_k[idx] += f1
        #     else:
        #         agglomerative_sums_f1s_for_k[idx] += f1

            axs.plot(k_values, task_f1s_for_k, marker=marker, label=clean_name(config.name) + f" - {task_idx}", color=color)

        # Calculate the average F1, precision, and recall for each configuration (dataset) averaging over k values
        # avg_f1 = np.mean(f1s_for_k)
        # std_f1 = np.std(f1s_for_k)
        # avg_prec = np.mean(precisions_for_k)
        # std_prec = np.std(precisions_for_k)
        # avg_rec = np.mean(recalls_for_k)
        # std_rec = np.std(recalls_for_k)
        # cluster_type = "Spectral" if "spectral" in config.name.lower() else "Agglomerative"
        # # print(f"{clean_name(config.name)}({cluster_type}) - Avg Prec: {avg_prec:.4f}\(\pm {std_prec:.4f}\), Avg Rec: {avg_rec:.4f}\(\pm {std_rec:.4f}\), Avg F1: {avg_f1:.4f}\(\pm {std_f1:.4f}\)")
        # print(f"{clean_name(config.name)}({cluster_type}) - "
        #     f"Avg Prec: {avg_prec:.4f} ± {std_prec:.4f}, "
        #     f"Avg Rec: {avg_rec:.4f} ± {std_rec:.4f}, "
        #     f"Avg F1: {avg_f1:.4f} ± {std_f1:.4f}")

        # if method == "macro":
        #     # ax.errorbar(k_values, f1s_for_k, yerr=f1_stds_for_k, fmt='o', alpha=0.5)
        #     ax.fill_between(k_values, [f1-std for f1, std in zip(f1s_for_k, f1_stds_for_k)], [f1+std for f1, std in zip(f1s_for_k, f1_stds_for_k)], alpha=0.2, color=ax.get_lines()[-1].get_color())
        #     #Plot the std deviation as a dotted line above and below the f1 scores keeping the same color as the f1 score line
        #     ax.plot(k_values, [f1+std for f1, std in zip(f1s_for_k, f1_stds_for_k)], linestyle='dotted', alpha=0.7, color=ax.get_lines()[-1].get_color())
        #     ax.plot(k_values, [f1-std for f1, std in zip(f1s_for_k, f1_stds_for_k)], linestyle='dotted', alpha=0.7, color=ax.get_lines()[-1].get_color())
        
        # else:
        #     ax.plot(k_values, f1s_for_k, marker='o', label=clean_name(config.name))

   
            axs.set_title(f'F1-{method.capitalize()} vs K - Agglomerative', fontsize=18)
            axs.tick_params(axis='both', labelsize=15)

            #Set scale of y axis to be from 0 to 1
            axs.set_ylim(0, 1)
            axs.set_xlim(k_values[0]-1, k_values[-1]+1)
            axs.set_xlabel('K', fontsize=17)
            axs.set_ylabel('F1 Score', fontsize=17)
            axs.grid(True)
            axs.legend(loc='best', fontsize=13)
    plt.tight_layout()
    plt.show()


    


    # # Display average F1 vs K for agglomerative and spectral separately
    # average_agglomerative_f1s_for_k = [s / (len(configs)/2) for s in agglomerative_sums_f1s_for_k]
    # average_spectral_f1s_for_k = [s / (len(configs)/2) for s in spectral_sums_f1s_for_k]
    # fig, axs = plt.subplots(1, 2, figsize=(12, 10))
    # fig.suptitle(f'Average F1-{method.capitalize()}')
    # axs[0].plot(k_values, average_agglomerative_f1s_for_k, marker='o', label='Agglomerative Average')
    # axs[0].set_title('Average F1 vs K - Agglomerative', fontsize=18)
    # axs[0].set_xlabel('K', fontsize=15)
    # axs[0].set_ylabel('Average F1 Score', fontsize=15)
    # axs[0].grid(True)
    # axs[0].legend(loc='best')
    # axs[0].tick_params(axis='both', labelsize=15)
    # axs[1].plot(k_values, average_spectral_f1s_for_k, marker='o', label='Spectral Average')
    # axs[1].set_title('Average F1 vs K - Spectral', fontsize=18)
    # axs[1].set_xlabel('K', fontsize=15)
    # axs[1].set_ylabel('Average F1 Score', fontsize=15)
    # axs[1].grid(True)
    # axs[1].legend(loc='best')
    # axs[1].tick_params(axis='both', labelsize=15)
    # plt.show()

    

    # # Display average F1 vs K across configurations
    # average_f1s_for_k = [s / len(configs) for s in sums_f1s_for_k]
    # plt.figure(figsize=(8, 6))
    # plt.title(f'Average F1 vs K across Configurations - {method.capitalize()}')
    # plt.plot(k_values, average_f1s_for_k, marker='o')
    # plt.xlabel('K', fontsize=15)
    # plt.ylabel('Average F1 Score', fontsize=15)
    # plt.grid(True)
    # plt.show()

if __name__ == '__main__':
    parser = ArgumentParser(description="Region Monitoring Test")
    parser.add_argument(
        '--params',
        type=str,
        help="/path/to/3cam_region_querying.yaml file of region monitoring tasks",
        default="config/field_tests/region_querying_3_cam.yaml"
    )
    args = parser.parse_args()
    
    with open(args.params, 'rb') as f:
        configs_3cam = list(yaml.safe_load_all(f))

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
            self.micro_f1_scores = []
            self.micro_precisions = []
            self.micro_recalls = []
            self.alphas = []
            self.ks = []
            self.best_f1 = 0.0
            self.best_alpha = 0.0
            self.best_precision = 0.0
            self.best_recall = 0.0
            self.macro_best_f1 = 0.0
            self.macro_best_alpha = 0.0
            self.macro_best_precision = 0.0
            self.macro_best_recall = 0.0
            self.macro_best_f1_std = 0.0
            self.macro_best_prec_std = 0.0
            self.macro_best_rec_std = 0.0
            self.best_k = 0
            self.macro_best_k = 0
            self.tasks = []
            self.place_nodes_dict = {}
            self.input_task_clip_embs = None
            self.input_task_clip_tensor = None
            self.task_metrics = []
            self.macro_precisions = []
            self.macro_recalls = []
            self.macro_f1s = []
            self.macro_f1_stds = []
            self.macrod_prec_stds = []
            self.macro_rec_stds = []

    configs = []
    for config_3cam in configs_3cam:

        terra_paths = config_3cam["terra"] # list of paths
        for tp in terra_paths:
            name = tp.split("nodist1img_")[1].split("cluster.pkl")[0]
            print(f"\nProcessing config: {name}")

            config = RegionQueryingConfig(
                name = name,
                terra_path=tp,
                region_tasks=config_3cam["region_tasks"],
                prediction_method=config_3cam["prediction_method"]
            )

            configs.append(config)

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
        config.input_task_clip_tensor.div_(config.input_task_clip_tensor.norm(dim=-1,keepdim=True))
        print(f"\nCollected region tasks for {config.name}:", config.tasks)

    

    # alpha_values = np.linspace(0.2, 0.35, 16)
    alpha_values = [0.26]
    k_values = np.linspace(1, 15, 15)
    # k_values = [4]


    # Find the best alpha and k based on evaluation metrics
    for config in configs:
        print(f"\nEvaluating configuration: {config.name}")

        #Get number of place nodes and region nodes
        print_number_of_place_and_region_nodes(config)


        for alpha in alpha_values:
            config.terra.alpha = alpha
            for k in k_values:
                if config.prediction_method == "aib": # and k != len(config.region_tasks):
                    k = len(config.region_tasks)  # Override k to be number of tasks for AIB since it doesn't vary with K
                    # continue  # Skip k values that don't match the number of tasks for AIB

                overall_region_nodes = [n for n, d in config.terra.terra_3dsg.nodes(data=True) if d["level"] > 1]
   
                if k > len(overall_region_nodes):
                    # Use the numbers from the previous k since we can't predict more regions than exist in the graph
                    config.micro_f1_scores.append(config.micro_f1_scores[-1])
                    config.micro_precisions.append(config.micro_precisions[-1])
                    config.micro_recalls.append(config.micro_recalls[-1])
                    config.alphas.append(alpha)
                    config.ks.append(k)
                    config.task_metrics.append(config.task_metrics[-1])
                    config.macro_precisions.append(config.macro_precisions[-1])
                    config.macro_recalls.append(config.macro_recalls[-1])
                    config.macro_f1s.append(config.macro_f1s[-1])
                    config.macro_f1_stds.append(config.macro_f1_stds[-1])
                    config.macrod_prec_stds.append(config.macrod_prec_stds[-1])
                    config.macro_rec_stds.append(config.macro_rec_stds[-1])                   
                    print(f"Skipping k={k} since it exceeds the number of available region nodes ({len(overall_region_nodes)})")
                    continue

                print(f"Predicting regions with alpha: {alpha:0.2f}, k: {k}")
                config.terra.reset_region_tasks()
                # Prediction regions given prompts
                config.terra.predict_regions(
                    config.input_task_clip_tensor, 
                    config.tasks, 
                    config.prediction_method,
                    K = int(k)
                )

                pred_places = config.terra.task_relevant_place_nodes

                micro_prec, micro_recall, f1, task_metrics, macro_prec, macro_rec, macro_f1, macro_prec_std, macro_rec_std, macro_f1_std = compute_region_metrics(pred_places, config.place_nodes_dict)
                config.micro_f1_scores.append(f1)
                config.micro_precisions.append(micro_prec)
                config.micro_recalls.append(micro_recall)
                config.alphas.append(alpha)
                config.ks.append(k)
                config.task_metrics.append(task_metrics)
                config.macro_precisions.append(macro_prec)
                config.macro_recalls.append(macro_rec)
                config.macro_f1s.append(macro_f1)
                config.macro_f1_stds.append(macro_f1_std)
                config.macrod_prec_stds.append(macro_prec_std)
                config.macro_rec_stds.append(macro_rec_std)
                print(f"Alpha: {alpha:0.2f}, K: {k} => Precision: {micro_prec:.4f}, Recall: {micro_recall:.4f}, F1: {f1:.4f}")
                
                # Update best parameters
                if f1 > config.best_f1:
                    config.best_f1 = f1
                    config.best_alpha = alpha
                    config.best_k = k
                    config.best_precision = micro_prec
                    config.best_recall = micro_recall

                if macro_f1 > config.macro_best_f1:
                    config.macro_best_f1 = macro_f1
                    config.macro_best_alpha = alpha
                    config.macro_best_k = k
                    config.macro_best_precision = macro_prec
                    config.macro_best_recall = macro_rec
                    config.macro_best_f1_std = macro_f1_std
                    config.macro_best_prec_std = macro_prec_std
                    config.macro_best_rec_std = macro_rec_std

        print(f"\nBest parameters found - Alpha: {config.best_alpha:0.2f}, K: {config.best_k} with Precision: {config.best_precision:.4f}, Recall: {config.best_recall:.4f}, F1: {config.best_f1:.4f}")



    # Find the best parameters for each task and print them out
    for config in configs:
        print(f"\n\n{config.name.upper()}:")
        tps, fps, fns = 0,0,0
        for task_idx, task in enumerate(config.region_tasks):
            best_task_tps, best_task_fps, best_task_fns = 0,0,0
            best_task_f1 = 0.0
            best_task_prec = 0.0
            best_task_rec = 0.0
            best_task_alpha = 0.0
            best_task_k = 0
            for alpha, k, task_metrics in zip(config.alphas, config.ks, config.task_metrics):
                task_prec, task_rec, task_f1, tp, fp, fn = task_metrics[task_idx]
                if task_f1 > best_task_f1:
                    best_task_f1 = task_f1
                    best_task_prec = task_prec
                    best_task_rec = task_rec
                    best_task_alpha = alpha
                    best_task_k = k
                    best_task_tps = tp
                    best_task_fps = fp
                    best_task_fns = fn
            print(f"  Task: {task['task']}")
            print(f"    Best Alpha: {best_task_alpha:0.2f}, Best K: {best_task_k} with Precision: {best_task_prec:.4f}, Recall: {best_task_rec:.4f}, F1: {best_task_f1:.4f}")

            tps += best_task_tps
            fps += best_task_fps
            fns += best_task_fns
        
        overall_prec, overall_rec, overall_f1 = compute_precision_recall_f1(tps, fps, fns)


        runtime_micro_ms = 0.0
        runtime_macro_ms = 0.0

        # #Get ms for best config
        # config.terra.reset_region_tasks()
        # config.terra.alpha = config.best_alpha
        # start_time = time.time()
        # config.terra.predict_regions(
        #     config.input_task_clip_tensor,
        #     config.tasks,
        #     config.prediction_method,
        #     K = int(config.best_k)
        # )
        # end_time = time.time()
        # runtime_micro_ms = (end_time - start_time)*1000

        # config.terra.reset_region_tasks()
        # config.terra.alpha = config.macro_best_alpha
        # start_time = time.time()
        # config.terra.predict_regions(
        #     config.input_task_clip_tensor,
        #     config.tasks,
        #     config.prediction_method,
        #     K = int(config.macro_best_k)
        # )
        # end_time = time.time()
        # runtime_macro_ms = (end_time - start_time)*1000

        print(f"---Overall Micro Best Metrics of alpha {config.best_alpha:0.2f} and K {int(config.best_k)} => Precision: {config.best_precision:.3f}, Recall: {config.best_recall:.3f}, F1: {config.best_f1:.3f}, Runtime: {runtime_micro_ms:.2f} ms---")
        print(f"---Overall Macro Best Metrics of alpha {config.macro_best_alpha:0.2f} and K {int(config.macro_best_k)} => Precision: {config.macro_best_precision:.3f} ± {config.macro_best_prec_std:.3f}, Recall: {config.macro_best_recall:.3f} ± {config.macro_best_rec_std:.3f}, F1: {config.macro_best_f1:.3f} ± {config.macro_best_f1_std:.3f}, Runtime: {runtime_macro_ms:.2f} ms---")

    
    # #Display ground truth place nodes for each task in the first config
    # first_config = configs[0]
    # print(f"\n\nDisplaying Ground Truth Place Nodes for {first_config.name.upper()}:")
    # for task_idx, place_nodes in first_config.place_nodes_dict.items():
    #     print(f"Task: {first_config.tasks[task_idx]}, Place Nodes: {place_nodes}")

    #     first_config.terra.visualizer.display_selected_nodes(
    #         first_config.terra.terra_3dsg,
    #         place_nodes,
    #         pc=first_config.terra.pc
    #     )


    # for config in configs:
    #     #Display 3dsg
    #     print(f"\n\nDisplaying results for {config.name.upper()}:")
    #     terra.display_3dsg()

    # #Plot grid of each 4 plots of F1 vs Alpha with best K for each configuration
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # i = 0
    # for config in configs:
    #     if config.prediction_method == "aib":
    #         continue  # Skip AIB for alpha plots since it doesn't vary with K
    #     ax = axs[i//2, i%2]
    #     best_ks = []
    #     best_f1s = []
    
    #     f1s_for_alpha = [f1 for a, k, f1 in zip(config.alphas, config.ks, config.f1_scores) if k == config.best_k]

    #     ax.plot(alpha_values, f1s_for_alpha, marker='o')
    #     ax.set_title(f'F1 vs Alpha - {config.name}(K={int(config.best_k)})')
    #     ax.set_xlabel('Alpha')
    #     ax.set_ylabel('F1 Score')
    #     ax.grid(True)

    #     i += 1
    
    # plt.show()


    plot_k_graphs(configs, k_values, method="micro")
    # plot_k_graphs(configs, k_values, method="macro")

    # #Plot grid of 4 plots of F1 vs K with best Alpha for each configuration
    # fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    # i = 0
    # sums_f1s_for_k = [0.0 for _ in k_values]
    # for config in configs:
    #     if config.prediction_method == "aib":
    #         continue  # Skip AIB for K plots since it doesn't vary with alpha
    #     ax = axs[i//2, i%2]
    #     best_ks = []
    #     best_f1s = []
    
    #     f1s_for_k = [f1 for a, k, f1 in zip(config.alphas, config.ks, config.f1_scores) if a == config.best_alpha]
    #     for k, f1 in zip(config.ks, config.f1_scores):
    #         idx = np.where(k_values == k)[0][0]
    #         sums_f1s_for_k[idx] += f1

    #     ax.plot(k_values, f1s_for_k, marker='o')
    #     ax.set_title(f'F1 vs K - {config.name}(Alpha={config.best_alpha:0.2f})')
    #     ax.set_xlabel('K')
    #     ax.set_ylabel('F1 Score')
    #     ax.grid(True)

    #     i += 1

    # plt.show()

    # average_f1s_for_k = [s / 2 for s in sums_f1s_for_k]
    # plt.figure(figsize=(8, 6))
    # plt.plot(k_values, average_f1s_for_k, marker='o')
    # plt.title('Average F1 vs K across Configurations')
    # plt.xlabel('K')
    # plt.ylabel('Average F1 Score')
    # plt.grid(True)
    # plt.show()

