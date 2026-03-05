import numpy as np
import open3d as o3d

from terra_eval.iou_helpers import IoU, Box
# from holo_bboxes import get_liosam2orig_transformation


def topk_bboxes_for_task(objs, task_idx, k):
    # Sort objects by their score for the given task index, descending
    sorted_objs = sorted(
        objs,
        key=lambda obj: obj.get_task_scores()[task_idx].item(),
        reverse=True
    )
    # Pick the top-k
    topk_objs = sorted_objs[:k]
    # Return their bounding boxes
    return [obj.get_bbox() for obj in topk_objs]

def compute_precision(gt_bboxes, pred_bboxes, map_gtid_2_idx, min_sim_ratio):
    '''
    Computes RPrec, SPrec
    based on Clio paper
    '''
    
    total_pred_objects = 0
    total_strict_matches = 0
    total_relaxed_matches = 0
    num_tasks = len(gt_bboxes.keys())
    max_task_scores = {task_idx: 0.0 for task_idx in range(num_tasks)}
    pred_task_objs = {task_idx: [] for task_idx in range(num_tasks)}
    for pred_obj in pred_bboxes:
        task_idx = pred_obj.get_task_idx()
        score = pred_obj.get_top_score()
        pred_task_objs[task_idx].append(pred_obj)
        if score > max_task_scores[task_idx]:
            max_task_scores[task_idx] = score
    
    for gt_id, gt_obbs in gt_bboxes.items():
        pred_obbs = []
        task_idx = map_gtid_2_idx[gt_id]
        for obj in pred_task_objs[task_idx]:
            score = obj.get_top_score()
            if score > (min_sim_ratio*max_task_scores[task_idx]):
                pred_obbs.append(obj.get_bbox())
        total_pred_objects += len(pred_obbs)
        
        matched_gt = set()
        matched_pred = set()
        while len(matched_gt) < len(gt_obbs) and len(matched_pred) < len(pred_obbs):
            best_iou = 0
            best_gt_pred_pair = None
            for gt_idx, gt_obb in enumerate(gt_obbs):
                if gt_idx in matched_gt:
                    continue  # skip already matched GT
                for pred_idx, pred_obb in enumerate(pred_obbs):
                    if pred_idx in matched_pred:
                        continue  # skip already matched Pred
                    iou = compute_iou_obb(gt_obb, pred_obb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_pred_pair = (gt_idx, pred_idx)
            if best_iou == 0:
                break
            
            best_gt_bbox = gt_obbs[best_gt_pred_pair[0]]
            best_pred_bbox = pred_obbs[best_gt_pred_pair[1]]
            if obb1_contains_obb2_centroid(best_pred_bbox, best_gt_bbox):
                total_relaxed_matches += 1
                if obb1_contains_obb2_centroid(best_gt_bbox, best_pred_bbox):
                    total_strict_matches += 1
            
            gt_idx, pred_idx = best_gt_pred_pair
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
    
    if total_pred_objects == 0:
        return 0, 0
    RPrec = total_relaxed_matches / total_pred_objects
    SPrec = total_strict_matches / total_pred_objects
    return RPrec, SPrec

def compute_acc_and_iou(gt_bboxes, pred_bboxes, map_gtid_2_idx):
    '''
    Computes RAcc, SAcc, and IoU
    based on Clio paper
    '''
    
    total_gt_objects = 0
    total_strict_matches = 0
    total_relaxed_matches = 0
    sum_iou = 0
    for gt_id, gt_obbs in gt_bboxes.items():
        N = len(gt_obbs) # number of ground truth bounding boxes
        total_gt_objects += N
        pred_idx = map_gtid_2_idx[gt_id]
        pred_obbs = topk_bboxes_for_task(pred_bboxes, pred_idx, N)
        
        matched_gt = set()
        matched_pred = set()
        while len(matched_gt) < N and len(matched_pred) < N:
            best_iou = 0
            best_gt_pred_pair = None
            for gt_idx, gt_obb in enumerate(gt_obbs):
                if gt_idx in matched_gt:
                    continue  # skip already matched GT
                for pred_idx, pred_obb in enumerate(pred_obbs):
                    if pred_idx in matched_pred:
                        continue  # skip already matched Pred
                    iou = compute_iou_obb(gt_obb, pred_obb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_pred_pair = (gt_idx, pred_idx)
            if best_iou == 0:
                break
                        
            sum_iou += best_iou
            best_gt_bbox = gt_obbs[best_gt_pred_pair[0]]
            best_pred_bbox = pred_obbs[best_gt_pred_pair[1]]
            if obb1_contains_obb2_centroid(best_pred_bbox, best_gt_bbox):
                total_relaxed_matches += 1
                if obb1_contains_obb2_centroid(best_gt_bbox, best_pred_bbox):
                    total_strict_matches += 1
                        
            gt_idx, pred_idx = best_gt_pred_pair
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
    
    RAcc = total_relaxed_matches / total_gt_objects
    SAcc = total_strict_matches / total_gt_objects
    iou = sum_iou / total_gt_objects
    return  RAcc, SAcc, iou


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

def fix_degenerate_obb(obb, eps=1e-6):
    if np.any(obb.extent < eps):
        new_extent = np.array(obb.extent)
        # Expand any axis with very small extent
        small_axes = new_extent < eps
        new_extent[small_axes] += eps
        obb.extent = new_extent
        return obb
    else:
        return obb

def get_box_verticies(obb):
    center = obb.get_center()
    verticies = np.take(obb.get_box_points(), [0,3,2,5,1,6,7,4], axis=0)
    stack_points =  np.vstack((center, verticies))
    return stack_points

def compute_iou_obb(obb1, obb2):
    fix_degenerate_obb(obb1)
    fix_degenerate_obb(obb2)
    
    box1 = Box(get_box_verticies(obb1))
    box2 = Box(get_box_verticies(obb2))
    iou = IoU(box1, box2)
    return iou.iou()
    # try:
    #     return iou.iou()
    # except:
    #     return iou.iou_sampling()
        
def point_in_obb(point, obb):
    """
    Check if a point is inside an OrientedBoundingBox.
    Works for both CPU and CUDA OBBs.
    """
    # Use the transpose of the rotation matrix for the inverse transform
    local_pt = obb.R.T @ (point - obb.center)
    half_extent = obb.extent / 2.0
    return np.all(np.abs(local_pt) <= half_extent)

def obb1_contains_obb2_centroid(obb1, obb2):
    """Check if obb1 contains the centroid of obb2."""
    return point_in_obb(obb2.center, obb1)