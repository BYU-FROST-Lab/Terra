import numpy as np
import networkx as nx
import torch

from terra.utils import tensor_cosine_similarity


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
        
    def predict(self, tasks_tensor, method="max", K=1):
        self.selected_placenodes = {}
        
        print("Method for region monitoring:", method)
        if method == "max":
            self._predict_max(tasks_tensor, K)
        elif method == "thresh":
            self._predict_thresh(tasks_tensor)
        elif method == "mix":
            self._predict_mix(tasks_tensor, K)
        elif method == "aib":
            self._predict_aib(tasks_tensor, K)
        else:
            print("Unrecognized method")
            exit()
        
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