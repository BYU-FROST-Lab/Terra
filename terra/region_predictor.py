import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt

from utils import tensor_cosine_similarity, chunked_tensor_cosine_similarity


class RegionPredictor:
    """
    Responsible for predicting Terra Regions according to task
    """
    
    def __init__(self, terra):
        self.terra = terra
        
        self.place_nodes = [n for n, d in self.terra.terra_3dsg.nodes(data=True) if d["level"] == 1]
        self.place_nodeid_to_idx = {n: i for i, n in enumerate(self.place_nodes)}
        self.place_embeddings = torch.vstack([self.terra.terra_3dsg.nodes[n]["embedding"] for n in self.place_nodes])
        
        self.region_nodes = [n for n, d in self.terra.terra_3dsg.nodes(data=True) if d["level"] > 1]
        self.region_nodeid_to_idx = {n: i for i, n in enumerate(self.region_nodes)}
        self.region_embeddings = torch.vstack([self.terra.terra_3dsg.nodes[n]["embedding"] for n in self.region_nodes])
        
        self.region_scores = None
        self.selected_placenodes = {}
        
    def predict(self, tasks_tensor, method="max", K=1, tasks_names=None, gt_place_nodes=None):
        self.selected_placenodes = {}
        score_comparison_results = None
        
        if tasks_names is not None:
            self.tasks_names = tasks_names

        print("Method for region monitoring:", method)
        if method == "max":
            self._predict_max(tasks_tensor, K)
        elif method == "thresh":
            self._predict_thresh(tasks_tensor)
        elif method == "mix":
            self._predict_mix(tasks_tensor, K)
        elif method == "aib":
            self._predict_aib(tasks_tensor, K)
        elif method == "test":
            score_comparison_results = self._predict_test(tasks_tensor, K, gt_place_nodes)
        elif method == "avg_diff":
            self._predict_avg_diff(tasks_tensor, K)
            # self._predict_test(tasks_tensor, K)
        else:
            print("Unrecognized method")
            exit()
        
        if score_comparison_results is not None:
            return self.selected_placenodes, score_comparison_results
        else:
            return self.selected_placenodes # {task_idx: set(place_node_ids), ...}
    
    def _predict_max(self, tasks_tensor, K):
        self.region_scores = tensor_cosine_similarity(
            self.region_embeddings, 
            tasks_tensor) # (num_region_nodes, num_tasks)

        for task_idx in range(self.region_scores.shape[1]):
            chosen_region_nodes = []
            topk_scores, topk_idxs = torch.topk(self.region_scores[:,task_idx], K)
            max_score = topk_scores.max().item()
            topk_nodes = [self.region_nodes[i] for i in topk_idxs.tolist()]
            chosen_region_nodes = topk_nodes
        
            # Descend through the tree to collect all child nodes
            selected = set()
            queue = chosen_region_nodes
            while queue:
                node = queue.pop()
                node_level = self.terra.terra_3dsg.nodes[node]["level"]
                if node_level == 1:
                    selected.add(node)
                # Explore children until reaching level 1
                for nbr in self.terra.terra_3dsg.neighbors(node):
                    nbr_level = self.terra.terra_3dsg.nodes[nbr]["level"]
                    if 1 <= nbr_level < node_level:
                        queue.append(nbr)
            self.selected_placenodes[task_idx] = selected
            
    def _predict_thresh(self, tasks_tensor):
        place_scores = tensor_cosine_similarity(
            self.place_embeddings, 
            tasks_tensor) # (num_place_nodes, num_tasks)
        
        self.region_scores = tensor_cosine_similarity(
            self.region_embeddings, 
            tasks_tensor) # (num_region_nodes, num_tasks)
        
        for task_idx in range(self.region_scores.shape[1]):
            max_score = self.region_scores[:, task_idx].max().item()
            mask = self.region_scores[:, task_idx] > self.terra.alpha
            above_threshold_idxs = torch.nonzero(mask, as_tuple=True)[0]
            chosen_region_nodes = [self.region_nodes[i] for i in above_threshold_idxs.tolist()]
            
            # Descend through the tree to collect all child nodes
            selected = set()
            queue = chosen_region_nodes
            while queue:
                node = queue.pop()
                node_level = self.terra.terra_3dsg.nodes[node]["level"]
                if node_level == 1:
                    p_idx = self.place_nodeid_to_idx[node]
                    if place_scores[p_idx,task_idx] > self.terra.alpha:
                        selected.add(node)    
                    
                # Explore children until reaching level 1
                for nbr in self.terra.terra_3dsg.neighbors(node):
                    nbr_level = self.terra.terra_3dsg.nodes[nbr]["level"]
                    if 1 <= nbr_level < node_level:
                        queue.append(nbr)
            self.selected_placenodes[task_idx] = selected
        
    def _predict_mix(self, tasks_tensor, K):
        place_scores = tensor_cosine_similarity(
            self.place_embeddings, 
            tasks_tensor) # (num_place_nodes, num_tasks)
        
        self.region_scores = tensor_cosine_similarity(
            self.region_embeddings, 
            tasks_tensor) # (num_region_nodes, num_tasks)

        for task_idx in range(self.region_scores.shape[1]):
            topk_scores, topk_idxs = torch.topk(self.region_scores[:,task_idx], K)
            max_score = topk_scores.max().item()
            topk_nodes = [self.region_nodes[i] for i in topk_idxs.tolist()]
            chosen_region_nodes = topk_nodes
            
            # Descend through the tree to collect all child nodes
            selected = set()
            queue = chosen_region_nodes
            while queue:
                node = queue.pop()
                node_level = self.terra.terra_3dsg.nodes[node]["level"]
                if node_level == 1:
                    p_idx = self.place_nodeid_to_idx[node]
                    if place_scores[p_idx,task_idx] > self.terra.alpha:
                        selected.add(node)
                # Explore children until reaching level 1
                for nbr in self.terra.terra_3dsg.neighbors(node):
                    nbr_level = self.terra.terra_3dsg.nodes[nbr]["level"]
                    if 1 <= nbr_level < node_level:
                        queue.append(nbr)
            self.selected_placenodes[task_idx] = selected

    def _predict_test(self, tasks_tensor, K, gt_place_nodes=None):
        place_scores = tensor_cosine_similarity(
            self.place_embeddings, 
            tasks_tensor) # (num_place_nodes, num_tasks)
        
        # # Plot histogram of place scores for each task to determine distribution
        # for task_idx in range(place_scores.shape[1]):
        #     scores = place_scores[:, task_idx].detach().cpu().numpy()
        #     plt.hist(scores, bins=50)
        #     plt.title(f"Histogram of Place Scores for Task {task_idx}")
        #     plt.xlabel("Cosine Similarity Score")
        #     plt.ylabel("Frequency")
        #     plt.grid(True)
        #     plt.show()
        
        self.region_scores = tensor_cosine_similarity(
            self.region_embeddings, 
            tasks_tensor) # (num_region_nodes, num_tasks)
        
        # print("Place Scores Shape:", place_scores.shape)
        # print("Region Scores Shape:", self.region_scores.shape)

        # Recursive function to get all descendant place nodes of a region node
        def get_descendant_place_nodes(region_node):
            descendant_place_nodes = set()
            children = list([nbr for nbr in self.terra.terra_3dsg.neighbors(region_node) if self.terra.terra_3dsg.nodes[nbr]["level"] < self.terra.terra_3dsg.nodes[region_node]["level"]])
            for child in children:
                if self.terra.terra_3dsg.nodes[child]["level"] == 1: # if child is place node, add to set
                    descendant_place_nodes.add(child)
                else: # if child is region node, get its descendants recursively
                    descendant_place_nodes.update(get_descendant_place_nodes(child))
            return descendant_place_nodes

        gt_region_nodes = {}
        # Find "gt_region_nodes" which are the parent region nodes whose the majority of it's descendents place nodes are in the gt_place_nodes for each task
        if gt_place_nodes is not None:
            for task_idx in range(place_scores.shape[1]):
                gt_places_for_task = gt_place_nodes[task_idx]
                gt_region_nodes_for_task = set()
                for node in self.region_nodes:
                    place_children = get_descendant_place_nodes(node)
                    place_children = [child for child in place_children if child in self.place_nodeid_to_idx]
                    if place_children:
                        num_gt_places = sum([1 for child in place_children if child in gt_places_for_task])
                        if num_gt_places / len(place_children) > 0.5: # if majority of place children are in gt, consider this region node as gt region node for this task
                            gt_region_nodes_for_task.add(node)
                gt_region_nodes[task_idx] = gt_region_nodes_for_task
                print(f"Task {task_idx}: Found {len(gt_region_nodes_for_task)} GT Region Nodes based on GT Place Nodes")

        # Calculate the percentage of region nodes with scores above all the scores of their children
        score_comparison_results = [] # list of tuples (task_idx, gt_percentage_higher, count_gt_region_higher, gt_count_total, non_gt_percentage_higher, count_non_gt_region_higher, non_gt_count_total)
        for task_idx in range(self.region_scores.shape[1]):
            region_scores = self.region_scores[:, task_idx].detach().cpu().numpy()
            place_scores_for_task = place_scores[:, task_idx].detach().cpu().numpy()
            count_gt_region_higher = 0
            count_non_gt_region_higher = 0
            gt_count_total = 0
            non_gt_count_total = 0
            for node in self.region_nodes:
                children = list([nbr for nbr in self.terra.terra_3dsg.neighbors(node) if self.terra.terra_3dsg.nodes[nbr]["level"] < self.terra.terra_3dsg.nodes[node]["level"]])
                child_scores = []
                if children and len(children) > 1:
                    for child in children:
                        if child in self.region_nodeid_to_idx: # region node children
                            child_scores.append(region_scores[self.region_nodeid_to_idx[child]])
                        elif child in self.place_nodeid_to_idx: # place node children
                            child_scores.append(place_scores_for_task[self.place_nodeid_to_idx[child]])
                    
                    if region_scores[self.region_nodeid_to_idx[node]] > max(child_scores):
                        if node in gt_region_nodes[task_idx]:
                            count_gt_region_higher += 1
                        else:
                            count_non_gt_region_higher += 1
                    gt_count_total += 1 if node in gt_region_nodes[task_idx] else 0
                    non_gt_count_total += 1 if node not in gt_region_nodes[task_idx] else 0

            gt_percentage_higher = count_gt_region_higher / gt_count_total if gt_count_total > 0 else 0
            non_gt_percentage_higher = count_non_gt_region_higher / non_gt_count_total if non_gt_count_total > 0 else 0
            print(f"Task {task_idx}: Percentage of GT Region Nodes with Scores Higher than All Children: {gt_percentage_higher:.4f} ({count_gt_region_higher}/{gt_count_total})")
            print(f"Task {task_idx}: Percentage of Non-GT Region Nodes with Scores Higher than All Children: {non_gt_percentage_higher:.4f} ({count_non_gt_region_higher}/{non_gt_count_total})")

            score_comparison_results.append((task_idx, gt_percentage_higher, count_gt_region_higher, gt_count_total, non_gt_percentage_higher, count_non_gt_region_higher, non_gt_count_total))

        
        # # Calculate average cosine similarity scores for region nodes children to see if there is a correlation between region scores and their children's scores
        # for task_idx in range(self.region_scores.shape[1]):
        #     region_scores = self.region_scores[:, task_idx].detach().cpu().numpy()
        #     place_scores_for_task = place_scores[:, task_idx].detach().cpu().numpy()
        #     score_diffs, avg_child_scores, node_scores = [], [], []
        #     for node in self.region_nodes:
        #         children = list([nbr for nbr in self.terra.terra_3dsg.neighbors(node) if self.terra.terra_3dsg.nodes[nbr]["level"] < self.terra.terra_3dsg.nodes[node]["level"]])
        #         child_scores = []
        #         # print(f"Node {node}(level {self.terra.terra_3dsg.nodes[node]['level']}) has children {children}(levels {[self.terra.terra_3dsg.nodes[child]['level'] for child in children]})")
        #         if children and len(children) > 1:
        #             for child in children:
        #                 if child in self.region_nodeid_to_idx: # region node children
        #                     child_scores.append(region_scores[self.region_nodeid_to_idx[child]])
        #                 elif child in self.place_nodeid_to_idx: # place node children
        #                     child_scores.append(place_scores_for_task[self.place_nodeid_to_idx[child]])
                    
        #             avg_child_score = np.mean(child_scores)
        #             node_score = region_scores[self.region_nodeid_to_idx[node]]
        #             score_diffs.append(node_score - avg_child_score)
        #             avg_child_scores.append(avg_child_score)
        #             node_scores.append(node_score)
            
        #     avg_diff = np.mean(score_diffs)
        #     std_diff = np.std(score_diffs)
        #     print(f"Task {task_idx}: Average Score Difference between Region Nodes and their Children's Average Scores: {avg_diff:.4f}", f"Std Dev: {std_diff:.4f}")

        #     # Plot histogram of score differences to see if region nodes tend to have higher scores than their children
        #     plt.hist(score_diffs, bins=25)
        #     #Plot average
        #     plt.axvline(avg_diff, color='red', linestyle='dashed', linewidth=1, label=f'Average Diff: {avg_diff:.4f}')
        #     # plt.axvline(avg_diff + std_diff, color='orange', linestyle='dashed', linewidth=1, label=f'Average + 1 Std Dev: {avg_diff + std_diff:.4f}')
        #     # plt.axvline(avg_diff - std_diff, color='orange', linestyle='dashed', linewidth=1, label=f'Average - 1 Std Dev: {avg_diff - std_diff:.4f}')
        #     plt.legend()

        #     if self.tasks_names is not None:
        #         plt.title(f"Histogram of Score Differences for Task {task_idx}: {self.tasks_names[task_idx]}")
        #     else:
        #         plt.title(f"Histogram of Score Differences for Task {task_idx}")
        #     plt.xlabel("Score Difference (Region - Average Child)")
        #     plt.ylabel("Frequency")
        #     plt.grid(True)
        #     plt.show()

        #     #Plot Scatter plot of region scores vs average child scores to see if there is a correlation
        #     line_of_best_fit = np.polyfit(avg_child_scores, node_scores, 1)
        #     correlation = np.corrcoef(avg_child_scores, node_scores)[0,1]
        #     x_vals = np.linspace(min(avg_child_scores), max(avg_child_scores), 100)
        #     plt.scatter(avg_child_scores, node_scores)
        #     plt.plot(x_vals, np.poly1d(line_of_best_fit)(x_vals), color='red', label=f'Line of Best Fit (correlation: {correlation:.2f})')
        #     plt.legend()
        #     if self.tasks_names is not None:
        #         plt.title(f"Region Scores vs Average Child Scores for Task {task_idx}: {self.tasks_names[task_idx]}")
        #     else:
        #         plt.title(f"Region Scores vs Average Child Scores for Task {task_idx}")
        #     plt.xlabel("Average Child Score")
        #     plt.ylabel("Region Node Score")
        #     plt.grid(True)
        #     plt.show()


        # Plot region nodes and their scores to see if there are any patterns in scores across levels
        for task_idx in range(self.region_scores.shape[1]):
            plt.figure(figsize=(8, 6))
            scores = self.region_scores[:, task_idx].detach().cpu().numpy()
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            cmap = plt.get_cmap("viridis")
            # colors = cmap(norm_scores)
            levels = np.array([self.terra.terra_3dsg.nodes[node]["level"] for node in self.region_nodes])
            gt_region_nodes_for_task = gt_region_nodes[task_idx]
            # Color region nodes based on whether they are gt region nodes or not
            colors = ['red' if node in gt_region_nodes_for_task else 'blue' for node in self.region_nodes]
            plt.scatter(scores, levels, c=colors, s=200)
            # Draw edges between region nodes to visualize tree structure, coloring edges based on scores of connected nodes
            for i, node in enumerate(self.region_nodes):
                for nbr in self.terra.terra_3dsg.neighbors(node):
                    if nbr in self.region_nodeid_to_idx: # only draw edges between region nodes
                        # Only draw edges between nodes of lower levels to higher levels to visualize tree structure
                        if self.terra.terra_3dsg.nodes[nbr]["level"] < self.terra.terra_3dsg.nodes[node]["level"]:
                            nbr_idx = self.region_nodeid_to_idx[nbr]
                            if scores[nbr_idx] < scores[i]:
                                plt.plot([scores[i], scores[nbr_idx]], [levels[i], levels[nbr_idx]], color="blue", alpha=0.4)
                            else:
                                plt.plot([scores[i], scores[nbr_idx]], [levels[i], levels[nbr_idx]], color="blue", alpha=0.4)
            
            if self.tasks_names is not None:    
                plt.title(f"Region Scores by Level", fontsize=18)
                print("Task:", self.tasks_names[task_idx])
            else:
                plt.title(f"Region Scores by Level for Task {task_idx}", fontsize=18)
            plt.ylabel("Region Node Level", fontsize=17)
            plt.xlabel("Cosine Similarity Score", fontsize=17)
            # Plot y only as integers since levels are discrete
            plt.yticks(np.arange(min(levels), max(levels)+1, 1), fontsize=15)
            plt.xticks(fontsize=15)
            plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Ground Truth Region Node', markerfacecolor='red', markersize=20), plt.Line2D([0], [0], marker='o', color='w', label='Non-Ground Truth Region Node', markerfacecolor='blue', markersize=20)], fontsize=13)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # for task_idx in range(self.region_scores.shape[1]):
        #     # topk_scores, topk_idxs = torch.topk(self.region_scores[:,task_idx], 15)
        #     topk_scores, topk_idxs = torch.topk(self.region_scores[:,task_idx], K)
        #     max_score = topk_scores.max().item()
        #     topk_nodes = [self.region_nodes[i] for i in topk_idxs.tolist()]
        #     chosen_region_nodes = topk_nodes

        #     # # Print out the levels of the chosen region nodes to verify they are > 1
        #     # print(f"Task {task_idx}: Chosen Region Nodes and Levels:")
        #     # for node in chosen_region_nodes:
        #     #     level = self.terra.terra_3dsg.nodes[node]["level"]
        #     #     score = self.region_scores[self.region_nodeid_to_idx[node], task_idx].item()
        #     #     print(f"  Node ID: {node}, Level: {level}, Score: {score:.4f}")
            
        #     # Descend through the tree to collect all child nodes
        #     selected = set()
        #     queue = chosen_region_nodes
        #     while queue:
        #         node = queue.pop()
        #         node_level = self.terra.terra_3dsg.nodes[node]["level"]
        #         if node_level == 1:
        #             p_idx = self.place_nodeid_to_idx[node]
        #             if place_scores[p_idx,task_idx] > self.terra.alpha:
        #                 selected.add(node)
        #         # Explore children until reaching level 1
        #         for nbr in self.terra.terra_3dsg.neighbors(node):
        #             nbr_level = self.terra.terra_3dsg.nodes[nbr]["level"]
        #             if 1 <= nbr_level < node_level:
        #                 queue.append(nbr)
        #     self.selected_placenodes[task_idx] = selected


        return score_comparison_results


    def _predict_avg_diff(self, tasks_tensor, K):
        place_scores = tensor_cosine_similarity(
            self.place_embeddings, 
            tasks_tensor) # (num_place_nodes, num_tasks)
        
        self.region_scores = tensor_cosine_similarity(
            self.region_embeddings, 
            tasks_tensor) # (num_region_nodes, num_tasks)

        for task_idx in range(self.region_scores.shape[1]):
            region_scores = self.region_scores[:, task_idx].detach().cpu().numpy()
            place_scores_for_task = place_scores[:, task_idx].detach().cpu().numpy()
            score_diffs = {} # node_idx: score_diff
            for node in self.region_nodes:
                children = list([nbr for nbr in self.terra.terra_3dsg.neighbors(node) if self.terra.terra_3dsg.nodes[nbr]["level"] < self.terra.terra_3dsg.nodes[node]["level"]])
                child_scores = []
                if children and len(children) > 1 and region_scores[self.region_nodeid_to_idx[node]] > self.terra.alpha: # only consider region nodes with scores above alpha and with multiple children to avoid noisy score diffs from leaf nodes or nodes with single child
                    for child in children:
                        if child in self.region_nodeid_to_idx: # region node children
                            child_scores.append(region_scores[self.region_nodeid_to_idx[child]])
                        elif child in self.place_nodeid_to_idx: # place node children
                            child_scores.append(place_scores_for_task[self.place_nodeid_to_idx[child]])
                if child_scores:
                    avg_child_score = np.mean(child_scores)
                    score_diff = region_scores[self.region_nodeid_to_idx[node]] - avg_child_score
                    node_idx = self.region_nodeid_to_idx[node]
                    score_diffs[node_idx] = score_diff
            
            if len(score_diffs) < K:
                print(f"Warning: Only {len(score_diffs)} region nodes with valid score differences for task {task_idx}, but K={K}. Reducing K to {len(score_diffs)}.")
                K = len(score_diffs)

            topk_idxs = sorted(score_diffs, key=score_diffs.get, reverse=True)[:K]
            chosen_region_nodes = [self.region_nodes[i] for i in topk_idxs]

            # Descend through the tree to collect all child nodes
            selected = set()
            queue = chosen_region_nodes
            while queue:
                node = queue.pop()
                node_level = self.terra.terra_3dsg.nodes[node]["level"]
                if node_level == 1:
                    p_idx = self.place_nodeid_to_idx[node]
                    if place_scores[p_idx,task_idx] > self.terra.alpha:
                        selected.add(node)
                # Explore children until reaching level 1
                for nbr in self.terra.terra_3dsg.neighbors(node):
                    nbr_level = self.terra.terra_3dsg.nodes[nbr]["level"]
                    if 1 <= nbr_level < node_level:
                        queue.append(nbr)
            self.selected_placenodes[task_idx] = selected







    def _predict_aib(self, tasks_tensor, K):
        place_scores = tensor_cosine_similarity(
            self.place_embeddings, 
            tasks_tensor) # (num_place_nodes, num_tasks)
        
        places_subgraph = self.terra.terra_3dsg.subgraph(self.place_nodes)
        aibcluster = AIBClustering(places_subgraph, place_scores, self.place_nodeid_to_idx, self.terra.alpha, K=K)
        self.selected_placenodes = aibcluster.run_aib()
        
'''
Acknowledgement to Clio repository for AIB implementation:
https://github.com/MIT-SPARK/Clio
'''
class AIBClustering():
    def __init__(self, places_graph, cos_sim_scores, map_nid2idx, alpha, K):
        self.places_graph = places_graph
        self.cos_sim = cos_sim_scores.detach().cpu().numpy() 
        self.map_nid2idx = map_nid2idx
        self.map_idx2nid = {idx: node_id for node_id, idx in map_nid2idx.items()}
        self.alpha = alpha
        self.topK = K
        self.delta = 1e-4
        
        self.p_y_given_x = None
        self.p_x = None
        self.p_y = None
        self.I_xy = None
        self.p_c_given_x = None
        self.d_Icy_weight = 1.0
        self.m = 0 # number of tasks (incl null task)
        self.n = 0 # number of place nodes
    
    def init_variables(self):
        ## Initialize Variables
        # 1) p(y|x)
        # 2) x_tilde (c for clusters instead of x_tilde) 
        # 3) p(x) = p(x_tilde) = 1/N (uniform dist)
        # 4) p(y|x_tilde) = p(y|x)
        num_tasks = self.cos_sim.shape[1]
        self.m = num_tasks + 1 # |Y| (incl null task)
        self.n = self.cos_sim.shape[0] # |X|
        K = min(self.m, self.topK)
        p_y_given_x_tmp = np.zeros((self.m, self.n)) # (num_tasks+1, num_pts)
        p_y_given_x_tmp[0,:] = self.alpha
        self.cos_sim = np.clip(self.cos_sim, 0, 1) # make sure no negative values
        p_y_given_x_tmp[1:,:] = self.cos_sim.T
        
        self.p_y_given_x = 1e-12 * np.ones((self.m, self.n))
        l = 1
        top_inds = np.argpartition(p_y_given_x_tmp, -1, axis=0)[-1:]
        null_tasks = np.where(top_inds==0)[1]
        while l <= K:
            top_inds = np.argpartition(p_y_given_x_tmp, -l, axis=0)[-l:]
            self.p_y_given_x[top_inds, np.arange(self.n)] += p_y_given_x_tmp[top_inds, np.arange(self.n)]
            l += 1
        self.p_y_given_x[:, null_tasks] = 1e-12
        self.p_y_given_x[0, null_tasks] = 1.0   
        self.p_x = np.ones(self.n) / self.n
        self.p_y = np.ones(self.m) / self.m
        self.p_y_given_x = self.p_y_given_x / np.sum(self.p_y_given_x, axis=0)
        
        # Compute mutual informationL I(X;Y) -> args: px,py,py_given_x
        self.I_xy = self.mutual_information(self.p_x, self.p_y, self.p_y_given_x)
                
        # similar to beta portion of equation 1 in paper
        self.d_Icy_weight = 1.0
        
        # find clusters (c == x_tilde in paper)
        self.p_c_given_x = np.eye(self.n)
        self.p_c = self.p_c_given_x @ self.p_x
        self.p_y_given_c = self.p_y_given_x * self.p_x @ np.transpose(self.p_c_given_x) / self.p_c
           
    def mutual_information(self,px,py,py_given_x):
        """Get mutual information between two PMFs."""
        px_ = px[px > 0]
        py_ = py[py > 0]
        py_given_x_ = py_given_x[py > 0, :]
        py_given_x_ = py_given_x_[:, px > 0]
        log_term = py_given_x_ / py_[:, None]
        log_term[log_term == 0] = 1  # this is to avoid infs
        log_term = np.log2(log_term)
        return np.sum((py_given_x_ * log_term) @ px_)
    
    def run_aib(self):
        self.init_variables()
                
        prev_Icy = self.mutual_information(self.p_c, self.p_y, self.p_y_given_c)
        last_merged_node = None
                
        t = 0
        delta = 0
        while True:
            if self.places_graph.number_of_edges() == 0:
                break
            
            update_edges = []
            if last_merged_node is None:
                update_edges = list(self.places_graph.edges())
            else:
                update_edges = [(last_merged_node, nb)
                                for nb in self.places_graph.neighbors(last_merged_node)]
            
            for e in update_edges:
                e0 = self.map_nid2idx[e[0]]
                e1 = self.map_nid2idx[e[1]]
                d = self.compute_edge_weight(self.p_c, self.p_y_given_c, e0, e1)
                self.places_graph[e[0]][e[1]]['weight'] = d   
            
            min_edge = min(self.places_graph.edges(),
                           key=lambda x: self.places_graph[x[0]][x[1]]["weight"])
            
            # update probabilities
            py_c_temp = np.copy(self.p_y_given_c)
            pc_temp = np.copy(self.p_c)
            pc_x_temp = np.copy(self.p_c_given_x)
            min_e0 = self.map_nid2idx[min_edge[0]]
            min_e1 = self.map_nid2idx[min_edge[1]]
            py_c_temp[:, min_e0] = (self.p_y_given_c[:, min_e0] * self.p_c[min_e0] +
                                    self.p_y_given_c[:, min_e1] * self.p_c[min_e1]) / \
                                    (self.p_c[min_e0] + self.p_c[min_e1])
            py_c_temp[:, min_e1] = 0

            pc_temp[min_e0] = self.p_c[min_e0] + self.p_c[min_e1]
            pc_temp[min_e1] = 0

            pc_x_temp[min_e0, :] = self.p_c_given_x[min_e0, :] + \
                self.p_c_given_x[min_e1, :]
            pc_x_temp[min_e1, :] = 0

            Icy = self.mutual_information(pc_temp, self.p_y, py_c_temp)
            dIcy = prev_Icy - Icy
            prev_Icy = Icy
            delta = self.d_Icy_weight * dIcy / self.I_xy
            if delta > self.delta:
                break
            
            self.p_y_given_c = py_c_temp
            self.p_c = pc_temp
            self.p_c_given_x = pc_x_temp
            
            last_merged_node = min_edge[0] # used by Clio to get edges/d_ij calc based on neighbors of last merged node
            self.places_graph = nx.contracted_edge(
                self.places_graph, min_edge, self_loops=False
            )
            t += 1
                
        ## Get Clusters from P(C|X)
        
        # if visualize:
        #     import seaborn as sns
        #     import matplotlib.pyplot as plt
        #     plt.figure(figsize=(10, 6))
        #     sns.heatmap(self.p_c_given_x, cmap="viridis", cbar=True)
        #     plt.xlabel("Original Cluster Indices")
        #     plt.ylabel("Clustered Indices")
        #     plt.title("Finished Probability Distribution of Tasks Given Clusters")
        #     plt.show()
            
        #     plt.figure(figsize=(10, 6))
        #     sns.heatmap(self.p_y_given_c, cmap="viridis", cbar=True)
        #     plt.xlabel("Cluster Index (num_clusters)")
        #     plt.ylabel("Task Index (num_tasks + null_task)")
        #     plt.title("Finished Probability Distribution of Tasks Given Clusters")
        #     plt.show()
        
        selected_places = {}
        for c_idx in range(self.p_y_given_c.shape[1]):
            task_idx_null = np.argmax(self.p_y_given_c[:,c_idx])
            if task_idx_null > 0:
                task_idx = task_idx_null - 1
                node_id = self.map_idx2nid[c_idx]
                if task_idx in selected_places:
                    selected_places[task_idx].append(node_id)
                else:
                    selected_places[task_idx] = [node_id]
        return selected_places
    
    def compute_edge_weight(self, p_c, p_y_given_c, nidx1, nidx2):
        prior = p_c[[nidx1, nidx2]] / np.sum(p_c[[nidx1, nidx2]])
        weight = (p_c[nidx1] + p_c[nidx2]) * self.js_divergence(p_y_given_c[:,[nidx1,nidx2]], prior)
        return weight
        
    def js_divergence(self, py_c, pc):
        """Compute Jensen-Shannon divergence between two PMFs."""
        assert py_c.shape[1] == pc.shape[0]
        joint_probs = py_c @ pc
        sum_entropies = np.sum(pc * self.shannon_entropy(py_c))
        return self.shannon_entropy(joint_probs) - sum_entropies
    
    def shannon_entropy(self, p):
        """Compute Shannon entropy of PMF."""
        return -np.sum(p * np.log2(p), axis=0)