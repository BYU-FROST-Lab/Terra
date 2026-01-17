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
                 pcidx_2_clipid: dict, # {global_pt_idx: {clip_id: count, ...}, ...}
                 clip_tensor: torch.Tensor, # (num_clip_ids, 512)
                 clip_tensor_semanticpc: torch.Tensor, # (num_semantic_pts, 512)
                 semantic_pc_idxs: list,
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
        self.pcidx_2_clipid = pcidx_2_clipid
        self.clip_tensor = clip_tensor
        self.clip_tensor_semanticpc = clip_tensor_semanticpc
        self.semantic_pc_idxs = semantic_pc_idxs
        self.dbscan = DBSCAN(eps=dbscan_params['eps'],
                             min_samples=dbscan_params['min_samples'])
        self.search_rad = search_rad
        self.terrain_thresh = terrain_thresh
        self.terrain_names = terrain_names
        self.num_terrain = len(terrain_names)
        self.alpha = alpha
        
        self.kdt = KDTree(self.pc)       
        
        distinct_colors = [[1,0,0],[0,1,0],[0,0,1],[1,0.5,0],[1,0,1], [0,1,1]]
        self.visualizer = TerraVisualizer(
            level_offset=50, 
            terrain_colors=distinct_colors[:len(self.terrain_names)]
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
    
    def predict_objects(self, tasks_tensor, task_names, method="ms_avg"):
        self.tasks.extend(task_names)
        self.prev_task_idx = len(self.tasks) - len(task_names)
        
        self.objects = self.object_predictor.predict(
            tasks_tensor,
            method
        )
        self.add_objects_to_3dsg()
    
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
    
    def nodes_above_level(self, min_level=1):
        level_dict = defaultdict(list)
        for n, d in self.terra_3dsg.nodes(data=True):
            if d["level"] > min_level:
                level_dict[d["level"]].append(n)
        return dict(level_dict)
    
        
    def display_places(self):
        self.visualizer.display_places(self.terra_3dsg)
    
    def display_regions(self):
        self.visualizer.display_regions(self.terra_3dsg)
    
    def display_task_relevant_places(self, task_idx=-1):
        prev_offset = self.visualizer.level_offset 
        self.visualizer.level_offset = 1
        if task_idx == -1:
            print("Displaying relevant places for all region monitoring tasks.")
            print("Tasks:", self.region_tasks)
            all_relevant_places = []
            for t_idx, place_nodes in self.task_relevant_place_nodes.items():
                all_relevant_places.extend(list(place_nodes))
            relevant_places_subgraph = self.terra_3dsg.subgraph(all_relevant_places)
            self.visualizer.display_3dsg(relevant_places_subgraph, pc=self.pc)
        else:
            print(f"Displaying relevant places for task: {self.region_tasks[task_idx]}")
            place_nodes = self.task_relevant_place_nodes[task_idx]
            relevant_places_subgraph = self.terra_3dsg.subgraph(place_nodes)
            self.visualizer.display_3dsg(relevant_places_subgraph, pc=self.pc)
        self.visualizer.level_offset = prev_offset
        
    def display_3dsg(self, display_pc=False):
        if display_pc:
            self.visualizer.display_3dsg(self.terra_3dsg, pc=self.pc)
        else:
            self.visualizer.display_3dsg(self.terra_3dsg)
    
    def display_terra(self, 
                      display_pc=False, 
                      plot_objects_on_ground=False,
                      color_pc_clip=True, 
                      color_terrain=False):
        self.visualizer.display_terra(self, display_pc, plot_objects_on_ground, color_pc_clip, color_terrain)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--terra',
                        type=str,
                        help='Filepath for Terra saved from build_terra.py')
    args = parser.parse_args()
    
    terra = load_terra(args.terra)
        
    # Display Regions 
    terra.display_regions()
    
    # Display Places
    terra.display_places()
    
    # Display full Terra
    terra.display_terra()
    terra.display_terra(display_pc=True, color_pc_clip=True, color_terrain=False)