from argparse import ArgumentParser
import yaml
import torch
import numpy as np

import clip
import open3d as o3d

from terra.utils import load_terra
from terra_eval.holoocean_bboxes import HoloBBoxes, get_liosam2orig_transformation
from terra_eval.sim_object_metrics import compute_precision, compute_acc_and_iou, copy_obb

def transform_gt_bboxes(gt_bboxes_dict, T_HOLO2LIO):
    gt_bboxes_xform_dict = {}
    for gt_id, gt_obbs in gt_bboxes_dict.items():
        gt_bboxes_xform_dict[gt_id] = []
        for obb in gt_obbs:
            obb_copy = copy_obb(obb)
            # obb_copy.transform(T_HOLO2LIO)
            obb_copy.center = T_HOLO2LIO[:3,:3] @ obb_copy.center + T_HOLO2LIO[:3,3]
            obb_copy.R = T_HOLO2LIO[:3,:3] @ obb_copy.R
            gt_bboxes_xform_dict[gt_id].append(obb_copy)
    return gt_bboxes_xform_dict

def create_wire_bbox(xmin, xmax, ymin, ymax, zmin, zmax, color=[1, 0, 0]):
    # 8 corners of the box
    points = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ])

    # 12 edges of the box
    lines = [
        [0,1], [1,2], [2,3], [3,0],  # bottom
        [4,5], [5,6], [6,7], [7,4],  # top
        [0,4], [1,5], [2,6], [3,7]   # vertical
    ]

    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, help="YAML file with object tasks and Terra path")
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load Terra
    terra = load_terra(cfg['terra'])
    terra.alpha = cfg['alpha']
    if 'bounds' in cfg:
        terra.object_predictor.xmin = cfg['bounds'][0]
        terra.object_predictor.xmax = cfg['bounds'][1]
        terra.object_predictor.ymin = cfg['bounds'][2]
        terra.object_predictor.ymax = cfg['bounds'][3]
        terra.object_predictor.zmin = cfg['bounds'][4]
        terra.object_predictor.zmax = cfg['bounds'][5]
        print("Updated bounds:",
              terra.object_predictor.xmin,
              terra.object_predictor.xmax,
              terra.object_predictor.ymin,
              terra.object_predictor.ymax,
              terra.object_predictor.zmin,
              terra.object_predictor.zmax)
        # Add bounds to plot
        wire_bbox = create_wire_bbox(
            terra.object_predictor.xmin,
            terra.object_predictor.xmax,
            terra.object_predictor.ymin,
            terra.object_predictor.ymax,
            terra.object_predictor.zmin,
            terra.object_predictor.zmax
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(terra.pc)
        # pcd.transform(get_liosam2orig_transformation(case))
        pcd.paint_uniform_color([0.5,0.5,0.5])
        o3d.visualization.draw_geometries([wire_bbox, pcd])
        
    print("Alpha parameter for object prediction:", terra.alpha)

    # Setup CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/16", device=device)

    # Encode tasks
    tasks = [task["task"] for task in cfg['object_tasks']]
    tasks[:0] = terra.terrain_names
    input_task_embs = [clip_model.encode_text(clip.tokenize([t]).to(device)).float() for t in tasks]
    input_task_tensor = torch.vstack(input_task_embs)
    input_task_tensor.div_(input_task_tensor.norm(dim=-1, keepdim=True))

    ## Get ground truth objects
    gt_ids = [tasks["label"] for tasks in cfg['object_tasks']]
    map_gtid_2_idx = {gt_id: idx for idx, gt_id in enumerate(gt_ids)}
    map_idx_2_gtid = {idx: gt_id for idx, gt_id in enumerate(gt_ids)}
    gt_bboxes = HoloBBoxes(cfg['gt_bboxes_csv'], cfg['gt_obj_names'], use_context=False)
    gt_bboxes.update_bbox_subset(gt_ids)
    gt_bboxes.display(terra.pc, case=cfg['case'],use_color=True, plot_subset=True)
    gt_bboxes_dict = gt_bboxes.get_gt_bboxes()
    ## Transform GT bboxes into point cloud frame
    T_LIO2HOLO = get_liosam2orig_transformation(cfg['case'])
    T_HOLO2LIO = np.eye(4)
    T_HOLO2LIO[:3,:3] = T_LIO2HOLO[:3,:3].T
    T_HOLO2LIO[:3,3] = - T_LIO2HOLO[:3,:3].T @ T_LIO2HOLO[:3,3]
    gt_bboxes_xform_dict = transform_gt_bboxes(gt_bboxes_dict, T_HOLO2LIO)

    # Predict objects
    terra.predict_objects(input_task_tensor, tasks[terra.num_terrain:], cfg['prediction_method'], cfg['trim'])
    pred_objects = terra.objects
    print(f"Predicted {len(pred_objects)} objects")
    
    # Display Terra
    terra.display_terra(display_pc=True, plot_objects_on_ground=True, color_pc_clip=False)
    
    # Compute evaluation metrics
    print("Computing Metrics...")
    RAcc, SAcc, iou = compute_acc_and_iou(gt_bboxes_xform_dict, pred_objects, map_gtid_2_idx)#, terra.pc)#, args.experiment_type)#, global_pc)
    # print(f"IoU = {iou}, SAcc = {SAcc}, RAcc = {RAcc}\n\n\n\n")
    RPrec, SPrec = compute_precision(gt_bboxes_xform_dict, pred_objects, map_gtid_2_idx, 0.9)#, terra.pc)#, args.experiment_type)#, global_pc)
    # print(f"SPrec = {SPrec}, RPrec = {RPrec}\n\n\n\n")
    F1 = 0.0 if (RPrec + RAcc) == 0 else 2 * (RPrec * RAcc) / (RPrec + RAcc)
    
    print(f"IoU = {iou:.3f}, SAcc = {SAcc:.3f}, RAcc = {RAcc:.3f}, SPrec = {SPrec:.3f}, RPrec = {RPrec:.3f}, F1 = {F1:.3f}\n\n\n\n")
    