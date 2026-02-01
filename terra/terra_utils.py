import open3d as o3d
import pickle as pkl
import numpy as np

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