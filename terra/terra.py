from argparse import ArgumentParser
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import numpy as np
import pickle as pkl
import torch
import networkx as nx
import open3d as o3d

from utils import tensor_cosine_similarity
from visualize_terra import Terra_Visualizer

def save_terra(terra, dest):
    if len(terra.objects) > 0:
        # Convert OBB to dictionaries to make it pickleable
        for tobj in terra.objects:
            obb = tobj.get_bbox()
            tobj.bbox = TerraOBB(np.asarray(obb.center),np.asarray(obb.R),np.asarray(obb.extent))
    with open(dest, "wb") as f:
        pkl.dump(terra, f)

def load_terra(src):
    with open(src, "rb") as f:
        terra = pkl.load(f)
    if len(terra.objects) > 0:
        # Convert OBB to dictionaries to make it pickleable
        for tobj in terra.objects:
            tobb = tobj.get_bbox()
            tobj.bbox = o3d.geometry.OrientedBoundingBox(tobb.center, tobb.R, tobb.extent)
    return terra


class TerraOBB():
    def __init__(self, center, R, extent):
        self.center = center
        self.R = R
        self.extent = extent


class TerraObject():
    def __init__(self, task_scores, obb, idx_offset=0):
        self.task_idx = task_scores.argmax().item() + idx_offset
        self.top_score = task_scores.max().item()
        self.task_scores = task_scores
        self.bbox = obb
    
    def get_task_idx(self):
        return self.task_idx
    
    def get_top_score(self):
        return self.top_score
    
    def get_task_scores(self):
        return self.task_scores
    
    def get_bbox(self):
        return self.bbox


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
        self.alpha = alpha
        
        self.kdt = KDTree(self.pc)       
        
        distinct_colors = [[1,0,0],[0,1,0],[0,0,1],[1,0.5,0],[1,0,1], [0,1,1]]
        self.visualizer = Terra_Visualizer(
            level_offset=50, 
            terrain_colors=distinct_colors[:len(self.terrain_names)]
        )
        
        # Init object and task parameters
        self.max_nid = max(self.terra_3dsg.nodes) if len(self.terra_3dsg.nodes) > 0 else 0
        self.objects = []
        self.objectidx_2_nodeid = {}
        self.nodeid_2_objectidx = {}
        self.tasks = []
        self.prev_task_idx = 0
    
    def display_places(self):
        self.visualizer.display_places(self.terra_3dsg)
    
    def display_regions(self):
        self.visualizer.display_regions(self.terra_3dsg)
    
    def display_3dsg(self, display_pc=False):
        if display_pc:
            self.visualizer.display_3dsg(self.terra_3dsg, pc=self.pc)
        else:
            self.visualizer.display_3dsg(self.terra_3dsg)
            

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
    
    # Display full 3DSG with point cloud
    terra.display_3dsg()
    terra.display_3dsg(display_pc=True)