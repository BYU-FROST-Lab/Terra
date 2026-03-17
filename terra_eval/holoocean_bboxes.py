import csv
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import yaml

class HoloBBoxes():
    def __init__(self, bbox_csv, obj_names, use_context=False):        
        with open(obj_names, 'rb') as f:
            obj_names = yaml.safe_load(f)['label_names']
        self.objid_map = self._build_object_id_map(obj_names)
        
        self.use_context = use_context
        if self.use_context:
            self.taskidx_map = {}
            for idx, (id, values) in enumerate(self.objid_map.items()):
                self.taskidx_map[idx] = values

        self.bboxes = self.read_bounding_boxes(bbox_csv)
        self.bboxes_o3d = self.convert_to_o3d_bboxes(self.bboxes)
        self.bboxes_o3d_color = self.convert_to_o3d_bboxes(self.bboxes, use_color=True)
        
        self.bboxes_subset = []
        self.bboxes_o3d_subset = []
        self.bboxes_o3d_color_subset = []
        return
    
    def get_gt_bboxes(self, get_color=False):
        gt_bboxes = {}
        if len(self.bboxes_o3d_subset) == 0:
            print("No subset chosen. Loading all bounding boxes")
            for bbox, bbox_o3d, bbox_o3d_color in zip(self.bboxes, self.bboxes_o3d, self.bboxes_o3d_color):
                bbox_id = bbox[-1]
                if bbox_id in gt_bboxes:
                    gt_bboxes[bbox_id].append(bbox_o3d if not get_color else bbox_o3d_color)
                else:
                    gt_bboxes[bbox_id] = [bbox_o3d if not get_color else bbox_o3d_color]
        else:
            print("Extracting subset bounding boxes")
            for bbox, bbox_o3d, bbox_o3d_color in zip(self.bboxes_subset, self.bboxes_o3d_subset, self.bboxes_o3d_color_subset):
                bbox_id = bbox[-1]
                if bbox_id in gt_bboxes:
                    gt_bboxes[bbox_id].append(bbox_o3d if not get_color else bbox_o3d_color)
                else:
                    gt_bboxes[bbox_id] = [bbox_o3d if not get_color else bbox_o3d_color]
        return gt_bboxes            
    
    def update_bbox_subset(self, id_list):
        self.bboxes_subset = []
        self.bboxes_o3d_subset = []
        self.bboxes_o3d_color_subset = []
        for bbox, bbox_o3d, bbox_o3d_color in zip(self.bboxes, self.bboxes_o3d, self.bboxes_o3d_color):
            bbox_id = bbox[-1]
            if bbox_id in id_list:
                self.bboxes_subset.append(bbox)
                self.bboxes_o3d_subset.append(bbox_o3d)
                self.bboxes_o3d_color_subset.append(bbox_o3d_color)
    
    def read_bounding_boxes(self, csv_path):
        """
        Reads bounding boxes from CSV. Expects roll, pitch, yaw in degrees.
        """
        boxes = []
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_id = int(row["ClassID"])
                if class_id not in [-1,0,29,32,1,17,23]: # [N/A, None, Unlabeled, Any, Asphalt, Grass, Sidewalk] 
                    origin = np.array([float(row["OriginX"]),
                                    float(row["OriginY"]),
                                    float(row["OriginZ"])])
                    extent = np.array([float(row["ExtentX"]),
                                    float(row["ExtentY"]),
                                    float(row["ExtentZ"])])
                    # Roll, Pitch, Yaw in degrees
                    roll = float(row["RotRoll_deg"])
                    pitch = float(row["RotPitch_deg"])
                    yaw = float(row["RotYaw_deg"])
                    if self.use_context:
                        boxes.append((origin, extent, roll, pitch, yaw, int(row["TaskID"])))
                    else:
                        boxes.append((origin, extent, roll, pitch, yaw, class_id))
        return boxes
        
    def convert_to_o3d_bboxes(self, bboxes, use_color=False):
        bboxes_o3d = []
        for origin_m, extent_m, roll, pitch, yaw, class_id in bboxes:
            if use_color:
                if self.use_context:
                    color = np.array(self.taskidx_map[class_id]["color"]) / 255.
                else:
                    color = np.array(self.objid_map[class_id]["color"]) / 255.
                bbox = self._create_color_oriented_bbox(origin_m, extent_m, roll, pitch, yaw, color=color)
            else:
                bbox = self._create_oriented_bbox(origin_m, extent_m, roll, pitch, yaw, color=[0,0,0])
            bboxes_o3d.append(bbox)
        return bboxes_o3d
    
    def display(self, global_pointcloud, case, use_color=False, plot_subset=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(global_pointcloud[:,:3])
        pcd.transform(get_liosam2orig_transformation(case))
        pcd.paint_uniform_color([0.5,0.5,0.5])
        if use_color:
            if plot_subset:
                o3d.visualization.draw_geometries(self.bboxes_o3d_color_subset+[pcd])
            else:
                o3d.visualization.draw_geometries(self.bboxes_o3d_color+[pcd])
        else:    
            if plot_subset:
                o3d.visualization.draw_geometries(self.bboxes_o3d_subset+[pcd])
            else:
                o3d.visualization.draw_geometries(self.bboxes_o3d+[pcd])
    
    def _build_object_id_map(self, obj_names):
        objid_map = {}
        for obj_dict in obj_names:
            objid_map[obj_dict["label"]] = {"name":obj_dict["name"], 
                                            "color":obj_dict["color"]}
        return objid_map
    
    def _create_color_oriented_bbox(self, center_RH_m, e_world_aabb_m, roll, pitch, yaw, color=(1, 0, 0)):
        """
        Creates a solid mesh for an oriented bounding box.
        """
        R_o3d = self._unreal_rot_to_o3d_R(roll, pitch, yaw)
        e_local = self._recover_local_half_extents_from_aabb(e_world_aabb_m, R_o3d)  # half-extents in local/OBB frame

        # Unreal gives half extents, Open3D expects full size
        full_size = e_local * 2
        full_size = np.maximum(full_size, 1e-4)

        # Create a box mesh and transform it
        box_mesh = o3d.geometry.TriangleMesh.create_box(*full_size)
        box_mesh.paint_uniform_color(color)

        # Move to center at origin
        box_mesh.translate(-box_mesh.get_center())

        # Apply rotation and translation
        box_mesh.rotate(R_o3d, center=(0, 0, 0))
        box_mesh.translate(center_RH_m)

        return box_mesh
    
    def _create_oriented_bbox(self, center_RH_m, e_world_aabb_m, roll, pitch, yaw, color=(1,0,0)):
        R_o3d = self._unreal_rot_to_o3d_R(roll, pitch, yaw)
        e_local = self._recover_local_half_extents_from_aabb(e_world_aabb_m, R_o3d)  # half-extents in local/OBB frame
        full_sizes = 2.0 * e_local
        obb = o3d.geometry.OrientedBoundingBox(center=center_RH_m, R=R_o3d, extent=full_sizes)
        obb.color = color
        return obb
    
    def _unreal_rot_to_o3d_R(self, roll_deg, pitch_deg, yaw_deg):
        """
        Unreal rotator -> rotation matrix, then convert LH->RH by flipping Y.
        We assume roll(X), pitch(Y), yaw(Z) degrees are already separated.
        """
        rpy_rad = np.radians([roll_deg, pitch_deg, yaw_deg])
        R_unreal = R.from_euler("xyz", rpy_rad).as_matrix()  # LH
        F = np.diag([1.0, -1.0, -1.0])                        # flip Y & Z
        R_o3d = F @ R_unreal @ F                             # RH
        return R_o3d

    def _recover_local_half_extents_from_aabb(self, extent_world_aabb, R_o3d):
        """
        Given world AABB half-extents (from get_actor_bounds) and the object's rotation matrix R,
        recover the object's local OBB half-extents e_local solving |R| * e_local = e_world.
        """
        A = np.abs(R_o3d)
        # Least-squares in case of near-singular configurations
        e_local, *_ = np.linalg.lstsq(A, extent_world_aabb, rcond=None)
        # Clamp tiny negatives from numerical issues
        e_local = np.maximum(e_local, 0.0)
        return e_local


def get_liosam2orig_transformation(case):
    # # T-stamp: 1.57
    # trans = np.array([3.4446244,0.54921293,8.49])
    # rot = R.from_quat([0.0,0.0,-0.2079117078310606,0.9781475971175166])
    # rot_mat = rot.as_matrix()
    # T = np.array([
    #     [rot_mat[0,0], rot_mat[0,1], rot_mat[0,2], trans[0]],
    #     [rot_mat[1,0], rot_mat[1,1], rot_mat[1,2], trans[1]],
    #     [rot_mat[2,0], rot_mat[2,1], rot_mat[2,2], trans[2]],
    #     [0, 0, 0, 1],
    # ])
    
    if case == "sparse":
        # Sparse13/SparseFull
        trans = np.array([5,-10,8.49]) # Terra Paper
        # trans = np.array([5,-9.57,8.49]) # Better
        rot_mat = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0,0,1]
        ])
        T = np.array([
            [rot_mat[0,0], rot_mat[0,1], rot_mat[0,2], trans[0]],
            [rot_mat[1,0], rot_mat[1,1], rot_mat[1,2], trans[1]],
            [rot_mat[2,0], rot_mat[2,1], rot_mat[2,2], trans[2]],
            [0, 0, 0, 1],
        ])
    elif case == "dense":
        # Dense18
        trans = np.array([5,-27,8.49])
        rot_mat = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        T = np.array([
            [rot_mat[0,0], rot_mat[0,1], rot_mat[0,2], trans[0]],
            [rot_mat[1,0], rot_mat[1,1], rot_mat[1,2], trans[1]],
            [rot_mat[2,0], rot_mat[2,1], rot_mat[2,2], trans[2]],
            [0, 0, 0, 1],
        ])
    else:
        print("Invalid case argument")
        exit()
    return T 