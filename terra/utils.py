import os
import re
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import pickle as pkl

# LEVEL = {
#     "OBJECT": 0,
#     "PLACE": 1,
#     "REGION": 2 # regions = 2+
# }

def tensor_cosine_similarity(emb1, emb2):
    unscaled_logit_cos_sim = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=2)
    return unscaled_logit_cos_sim  

def chunked_tensor_cosine_similarity(
    emb1,               # (N, D)
    emb2,               # (M, D)
    chunk_size=8192,
):
    """
    Yields (start_idx, cosine_similarity_chunk)
    cosine_similarity_chunk shape: (chunk, M)
    """
    N = emb1.shape[0]
    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            emb1_chunk = emb1[start:end]  # (chunk, D)

            # Same operation as original, just chunked
            cos_sim = F.cosine_similarity(
                emb1_chunk.unsqueeze(1),  # (chunk, 1, D)
                emb2.unsqueeze(0),        # (1, M, D)
                dim=2
            )  # (chunk, M)

            yield start, cos_sim


def numeric_key(path):
        match = re.search(r"(\d+\.\d+)", path.stem)  # grabs the float in the name
        return float(match.group()) if match else float('inf')

def int_defaultdict():
    return defaultdict(int)

def find_latest_itr(folder: str):
    """Finds the largest iteration number."""
    regex = re.compile(r"clip_segs_itr(\d+)\.pt")
    max_itr = -1
    for f in os.listdir(folder):
        match = regex.search(f)
        if match:
            itr = int(match.group(1))
            if itr > max_itr:
                max_itr = itr
    return max_itr

def find_latest_file(folder: str, pattern=r"clip_segs_itr(\d+)\.pt"):
    """Finds the clip_segs_itr{last_itr}.pt file with the largest iteration number."""
    regex = re.compile(pattern)
    max_itr, latest_file = -1, None
    for f in os.listdir(folder):
        match = regex.search(f)
        if match:
            itr = int(match.group(1))
            if itr > max_itr:
                max_itr = itr
                latest_file = f
    return latest_file

def random_color():
    return np.random.rand(3).tolist()  # Generates a random RGB color


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

def copy_obb(original_obb):
    """
    Creates a new OBB object that is a deep copy of the original
    using the correct Open3D constructor.
    """
    # Create a new OBB object by explicitly calling the constructor
    # with the properties of the original OBB.
    return o3d.geometry.OrientedBoundingBox(
        original_obb.center,
        original_obb.R,
        original_obb.extent
    )

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