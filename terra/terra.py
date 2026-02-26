from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import torch
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import networkx as nx

from visualize_terra import TerraVisualizer
from object_predictor import ObjectPredictor
from region_predictor import RegionPredictor
from terra_utils import TerraObject, TerraOBB, load_terra


class Terra():
    def __init__(self,
                 terra_3dsg: nx.Graph,
                 pc: np.array, # global point cloud (num_pts, 3)
                 nodeid_2_imgidx: dict, # {node_id: [img_idx0, img_idx1, ...], }
                 image_names: list,
                 gidx_2_clipcounts: dict, # {global_pt_idx: {clip_id: count, ...}, ...}
                 clip_segs: torch.Tensor, # (num_clip_ids, 512)
                 semantic_gidx_avgclip: torch.Tensor, # (num_semantic_pts, 512)
                 semantic_gidxs: list,
                 dbscan_params: dict,
                 search_rad: float,
                 terrain_thresh: float,
                 terrain_names: list,
                 alpha: float,
                 ):
        self.terra_3dsg = terra_3dsg
        self.pc = pc
        self.nodeid_2_img_idx = nodeid_2_imgidx
        self.img_names = image_names
        self.gidx_2_clipcounts = gidx_2_clipcounts
        self.clip_segs = clip_segs
        self.semantic_gidx_avgclip = semantic_gidx_avgclip
        self.semantic_gidxs = semantic_gidxs
        self.dbscan = DBSCAN(eps=dbscan_params['eps'],
                             min_samples=dbscan_params['min_samples'])
        self.search_rad = search_rad
        self.terrain_thresh = terrain_thresh
        self.terrain_names = terrain_names
        self.num_terrain = len(terrain_names)
        self.alpha = alpha
        
        self.kdt = KDTree(self.pc)       
        
        self.visualizer = TerraVisualizer(
            level_offset=50, 
            num_terrains=len(self.terrain_names)
        )
        self.object_predictor = ObjectPredictor(self)
        self.region_predictor = RegionPredictor(self)
        
        # Init object and task parameters
        self.max_nid = max(self.terra_3dsg.nodes) if len(self.terra_3dsg.nodes) > 0 else 0
        ## For object retrieval
        self.objects = []
        self.objectidx_2_nodeid = {}
        self.nodeid_2_objectidx = {}
        self.tasks = []        
        self.prev_task_idx = 0
        ## For region monitoring
        self.region_tasks = []
        self.prev_region_task_idx = 0
        self.task_relevant_place_nodes = {}
        ## For path planning
        self.path_node_list = []
        self.start_node = 0
        self.dest_node = -1
    
    def predict_objects(self, tasks_tensor, task_names, method="ms_avg"):
        self.tasks.extend(task_names)
        self.prev_task_idx = len(self.tasks) - len(task_names)
        
        self.objects = self.object_predictor.predict(
            tasks_tensor,
            method
        )
        print(f"Now adding {len(self.objects)} objects to 3DSG")
        self.add_objects_to_3dsg()
        print(f"Finished adding {len(self.objects)} objects to 3DSG")
    
    def add_objects_to_3dsg(self):
        place_nodes = [n for n, d in self.terra_3dsg.nodes(data=True) if d["level"] == 1]
        place_pos = np.array([self.terra_3dsg.nodes[n]["pos"] for n in place_nodes])
        kdt = KDTree(place_pos)
        for i, tobj in enumerate(self.objects):
            self.max_nid = self.max_nid + 1
            obb_center = tobj.get_bbox().center[:2]
            self.terra_3dsg.add_node(
                self.max_nid,
                level=0,
                pos=tobj.get_bbox().center[:], # objects position = xyz
                terrain_id=-1,
            )
            self.objectidx_2_nodeid[i] = self.max_nid
            # Add edge to nearest place node
            dist, idx = kdt.query(obb_center)
            closest_place_node = place_nodes[idx]
            self.terra_3dsg.add_edge(self.max_nid, closest_place_node)
        self.nodeid_2_objectidx = {nid: obj_idx for obj_idx, nid in self.objectidx_2_nodeid.items()}
        
    def predict_regions(self, tasks_tensor, task_names, method="max", K=1):
        self.region_tasks.extend(task_names)
        self.prev_region_task_idx = len(self.region_tasks) - len(task_names)
        
        pred_place_nodes = self.region_predictor.predict(
            tasks_tensor,
            method,
            K
        )
        self.update_task_relevant_place_nodes(pred_place_nodes)
    
    def update_task_relevant_place_nodes(self, pred_place_nodes):
        for curr_task_idx, place_nodes in pred_place_nodes.items():
            task_idx = self.prev_region_task_idx + curr_task_idx
            self.task_relevant_place_nodes[task_idx] = place_nodes

    def plan_path_to_destination(self, 
                            task_tensor, 
                            terrain_preferences, 
                            method="ms_avg",
                            start_node=None):
        self.path_node_list = []
        
        place_nodes = [n for n, d in self.terra_3dsg.nodes(data=True) if d["level"] == 1]
        place_subgraph = self.terra_3dsg.subgraph(place_nodes)
        
        if task_tensor.shape[0] == self.num_terrain + 2:
            self.start_node = self._select_best_place_node(
                task_tensor[:-1,:], # skip destination query embedding
                method
            )
            mask = torch.ones(task_tensor.size(0), dtype=torch.bool, device=task_tensor.device)
            mask[-2] = False # remove source query embedding
            task_tensor_excluded = task_tensor[mask]   # shape: (N-1, 512)
            self.dest_node = self._select_best_place_node(
                task_tensor_excluded,
                method
            )
        else:
            self.start_node = place_nodes[0] if start_node is None else start_node
            self.dest_node = self._select_best_place_node(
                task_tensor,
                method
            )
        
        terrain_weight = self._make_terrain_weight(
            preferred=terrain_preferences.get("preferred", None),
            forbidden=terrain_preferences.get("forbidden", None),
            penalties=terrain_preferences.get("penalties", None)
        )
        self.path_node_list = nx.astar_path(
            place_subgraph, 
            source=self.start_node, 
            target=self.dest_node, 
            weight=terrain_weight
        )
    
    def _select_best_place_node(self, task_tensor, method="ms_avg"):
        pos_destination_objects = self.object_predictor.predict(
            task_tensor,
            method
        )
        max_dest_score = 0.0
        max_dest_obj_idx = -1
        for i, tobj in enumerate(pos_destination_objects):
            if tobj.get_top_score() > max_dest_score:
                max_dest_score = tobj.get_top_score()
                max_dest_obj_idx = i
        chosen_obb = pos_destination_objects[max_dest_obj_idx].get_bbox()
        obb_center = chosen_obb.center[:2]
        place_nodes = [n for n, d in self.terra_3dsg.nodes(data=True) if d["level"] == 1]
        place_pos = np.array([self.terra_3dsg.nodes[n]["pos"] for n in place_nodes])
        kdt = KDTree(place_pos)
        dist, idx = kdt.query(obb_center)
        
        best_place_node = place_nodes[idx]
        return best_place_node
        
    def _make_terrain_weight(self, preferred=None, forbidden=None, penalties=None):
        """
        Returns a weight function for A* that accounts for terrain types.
        
        Args:
            G : networkx.Graph
            preferred : set of terrain_ids that should have no penalty (default: None)
            forbidden : set of terrain_ids that should not be used (default: None)
            penalties : dict mapping terrain_id -> penalty cost (default: None)
        """
        preferred = preferred or set()
        forbidden = forbidden or set()
        penalties = penalties or {}

        def terrain_weight(u, v, d):
            base = d.get("weight", 1.0)
            t_u = self.terra_3dsg.nodes[u].get("terrain_id")
            t_v = self.terra_3dsg.nodes[v].get("terrain_id")

            # Block forbidden terrains entirely
            if t_u in forbidden or t_v in forbidden:
                return float("inf")  # edge not traversable

            # Apply penalties if terrain has one
            penalty = penalties.get(t_u, 0) + penalties.get(t_v, 0)

            # Neutral if in preferred set (penalty 0)
            if t_u in preferred and t_v in preferred:
                penalty = 0

            return base + penalty

        return terrain_weight
    
    def nodes_above_level(self, min_level=1):
        level_dict = defaultdict(list)
        for n, d in self.terra_3dsg.nodes(data=True):
            if d["level"] > min_level:
                level_dict[d["level"]].append(n)
        return dict(level_dict)
         
    def display_places(self, display_pc=False, plot_ids=False, no_spheres=False):
        if display_pc:
            self.visualizer.display_places(self.terra_3dsg, pc=self.pc, plot_ids=plot_ids, no_spheres=no_spheres)
        else:
            self.visualizer.display_places(self.terra_3dsg, plot_ids=plot_ids, no_spheres=no_spheres)
    
    def display_regions(self):
        self.visualizer.display_regions(self.terra_3dsg)
            
    def display_3dsg(self, display_pc=False):
        if display_pc:
            self.visualizer.display_3dsg(self.terra_3dsg, pc=self.pc)
        else:
            self.visualizer.display_3dsg(self.terra_3dsg)
    
    def display_terra(self, 
                      display_pc=False, 
                      plot_objects_on_ground=False,
                      color_pc_clip=True, 
                      color_terrain=False,
                      plot_ids=False):
        self.visualizer.display_terra(self, display_pc, plot_objects_on_ground, color_pc_clip, color_terrain, plot_ids=plot_ids)

    def display_task_relevant_places(self, task_idx=-1, heatmap_mode=False):
        self.visualizer.display_task_relevant_places(
            self.terra_3dsg,
            self.region_tasks,
            self.task_relevant_place_nodes,
            self.pc,
            task_idx,
            self.region_predictor.region_scores if heatmap_mode else None
        )
        
    def display_path(self):
        self.visualizer.display_path(
            self.terra_3dsg,
            self.path_node_list, 
            self.pc
        )

    def reset_region_tasks(self):
        self.region_tasks = []
        self.task_relevant_place_nodes = {}
        self.prev_region_task_idx = 0

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--Terra',
                        type=str,
                        helpk='Filepath for Terra saved from build_terra.py')
    args = parser.parse_args()
    
    terra = load_terra(args.terra)
        
    # Display Regions 
    terra.display_regions()
    
    # Display Places
    terra.display_places()
    
    # Display full Terra
    terra.display_terra()
    terra.display_terra(display_pc=True, color_pc_clip=True, color_terrain=False)