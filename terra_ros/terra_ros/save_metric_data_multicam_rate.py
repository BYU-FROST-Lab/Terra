#!/usr/bin/env python3
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py.point_cloud2 import read_points
import tf2_ros
from rclpy.duration import Duration


'''
Class for: 
    1) Subscribing to the Local PC, Global PC, time-synced Camera Images, and Path Nodes
    2) Compute transformations from Local PC to Global PC & from Camera Image to Global PC (using path nodes)
    3) Save each transformation, Local PC, Camera Image, and the ending Global PC Map
'''
class SaveMetricDataMultiCam(Node):
    def __init__(self):
        super().__init__('save_metric_data_multicam_rate')
        
        self.get_logger().info("Saving metric data")
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('save_folder', rclpy.Parameter.Type.STRING),
                ('save_period', rclpy.Parameter.Type.DOUBLE),
                ('lidar_topic', rclpy.Parameter.Type.STRING),
                ('num_cameras', rclpy.Parameter.Type.INTEGER),
                ('publish_extrinsics', rclpy.Parameter.Type.BOOL),
            ]
        )
                
        # Get parameters
        save_folder = self.get_parameter('save_folder').get_parameter_value().string_value
        self.save_period = self.get_parameter('save_period').get_parameter_value().double_value
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.publish_extrinsics = self.get_parameter('publish_extrinsics').get_parameter_value().bool_value
        
        camera_topics = []
        translations = []
        rotations = []
        parent_frames = []
        child_frames = []
        for nc in range(self.num_cameras):
            # Camera topic
            self.declare_parameter(f'camera{nc+1}_topic', '')
            camera_topics.append(
                self.get_parameter(f'camera{nc+1}_topic').get_parameter_value().string_value
            )
            
            # Extrinsics (only if enabled)
            if self.publish_extrinsics:
                self.declare_parameter(f'lidar_to_camera{nc+1}.translation', [0.0, 0.0, 0.0])
                self.declare_parameter(f'lidar_to_camera{nc+1}.rotation_quaternion', [0.0, 0.0, 0.0, 1.0])
                self.declare_parameter(f'lidar_to_camera{nc+1}.parent_frame', '')
                self.declare_parameter(f'lidar_to_camera{nc+1}.child_frame', '')
            
                translations.append(
                    self.get_parameter(f'lidar_to_camera{nc+1}.translation').get_parameter_value().double_array_value
                )
                rotations.append(
                    self.get_parameter(f'lidar_to_camera{nc+1}.rotation_quaternion').get_parameter_value().double_array_value
                )
                parent_frames.append(
                    self.get_parameter(f'lidar_to_camera{nc+1}.parent_frame').get_parameter_value().string_value
                )
                child_frames.append(
                    self.get_parameter(f'lidar_to_camera{nc+1}.child_frame').get_parameter_value().string_value
                )
            
        # Log for debugging
        self.get_logger().info(f'save_folder: {save_folder}')
        self.get_logger().info(f'lidar_topic: {lidar_topic}')
        self.get_logger().info(f'publish_extrinsics: {self.publish_extrinsics}')
        for nc in range(self.num_cameras):
            self.get_logger().info(f'camera_topics: {camera_topics[nc]}')
            self.get_logger().info(f'translation: {translations[nc]}')
            self.get_logger().info(f'rotation_quaternion: {rotations[nc]}')
            self.get_logger().info(f'{parent_frames[nc]} -> {child_frames[nc]}')        
        
        for nc in range(self.num_cameras):
            self.create_subscription(
                Image,
                camera_topics[nc],
                lambda msg, cam_id=nc: self.cam_img_callback(msg, cam_id),
                10,
            )
        self.lidar_pc_sub = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.lidar_pc_callback,
            10,
        )
        self.global_pc_sub = self.create_subscription(
            PointCloud2,
            '/lio_sam/mapping/map_global',
            self.global_pc_callback,
            10
        )
        self.path_sub = self.create_subscription(
            Path,
            '/lio_sam/mapping/path',
            self.path_callback,
            10
        )
        self.total_path_length = 0.0
        self.prev_position = None
        
        if self.publish_extrinsics:
            self.trans_l2c = []
            self.quat_l2c = []
            for nc in range(self.num_cameras):
                self.trans_l2c.append(np.array(translations[nc])) # [x,y,z]
                self.quat_l2c.append(np.array(rotations[nc]))  # [qx, qy, qz, qw]

        self.base_dir = save_folder
        os.makedirs(self.base_dir, exist_ok=True)
        self.lidar_pc_dir = os.path.join(self.base_dir, 'lidar_pc')
        os.makedirs(self.lidar_pc_dir, exist_ok=True)
        self.global_pc_dir = os.path.join(self.base_dir, 'global_pc')
        os.makedirs(self.global_pc_dir, exist_ok=True)
        self.cam_imgs_dirs = []
        self.trans_l2c_pc_dirs = []
        for nc in range(self.num_cameras):
            self.cam_imgs_dirs.append(os.path.join(self.base_dir, f'camera{nc+1}_images'))
            os.makedirs(self.cam_imgs_dirs[-1], exist_ok=True)
            self.trans_l2c_pc_dirs.append(os.path.join(self.base_dir, f'transformations_lidar2cam{nc+1}'))
            os.makedirs(self.trans_l2c_pc_dirs[-1], exist_ok=True)
        self.trans_l2g_pc_dir = os.path.join(self.base_dir, 'transformations_lidar2global')
        os.makedirs(self.trans_l2g_pc_dir, exist_ok=True)
        
        self.last_save_time = None # seconds 
        self.cam_buffer = [[] for _ in range(self.num_cameras)]
        self.lidar_buffer = []
        self.max_dt = 0.05          # [sec] max allowed camera–lidar skew
        # self.lookback = 0.5  # [sec] grab data this long ago 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.timer = self.create_timer(self.save_period, self.save_callback)
        # self.cam_tstamps = [[] for _ in range(self.num_cameras)]
        # self.lidar_tstamps = []
        # self.lidar_camera_pairs = []

    def path_callback(self, msg):
        if not msg.poses:
            return

        # Get latest pose (LIO-SAM appends to the end)
        pose = msg.poses[-1].pose.position
        current_position = np.array([pose.x, pose.y, pose.z])

        if self.prev_position is not None:
            step_dist = np.linalg.norm(current_position - self.prev_position)
            self.total_path_length += step_dist

            if step_dist > 0.0:
                self.get_logger().info(
                    f"\nTrajectory length: {self.total_path_length:.3f} m "
                    f"(+{step_dist:.3f} m)\n"
                )

        self.prev_position = current_position
    
    def cam_img_callback(self, msg, cam_id):
        t = self.header_to_seconds(msg.header)
        img_data = np.frombuffer(msg.data, dtype=np.uint8)
        img = img_data.reshape(msg.height, msg.width, -1)

        self.cam_buffer[cam_id].append((t, img))
        # self.cam_tstamps[cam_id].append(t)
        
    def lidar_pc_callback(self, msg):
        t = self.header_to_seconds(msg.header)
        self.lidar_buffer.append((t, msg))
        # self.lidar_tstamps.append(t)
        
    def save_callback(self):
        if not self.lidar_buffer:
            return
        
        # Need at least one image from every camera
        if any(len(buf) == 0 for buf in self.cam_buffer):
            return
        
        best = None  # (score, lidar_time, lidar_msg, cam_matches)
        for lidar_time, lidar_msg in self.lidar_buffer:
            cam_matches = []
            max_dt = 0.0
            for cam_id in range(self.num_cameras):
                t_img, img = min(
                    self.cam_buffer[cam_id],
                    key=lambda x: abs(x[0] - lidar_time)
                )
                dt = abs(t_img - lidar_time)
                max_dt = max(max_dt, dt)
                cam_matches.append((cam_id, t_img, img))

            if max_dt > self.max_dt:
                continue
            if best is None or max_dt < best[0]:
                best = (max_dt, lidar_time, lidar_msg, cam_matches)
        if best is None:
            self.cam_tstamps = [[] for _ in range(self.num_cameras)]
            self.lidar_tstamps = []
            self.lidar_camera_pairs = []
            self.lidar_buffer = []
            for cam_id in range(self.num_cameras):
                self.cam_buffer[cam_id] = []
            return
        
        _, lidar_time, lidar_msg, cam_matches = best
            
        try:
            tf_l2g = self.tf_buffer.lookup_transform(
                'map',
                lidar_msg.header.frame_id,
                Time.from_msg(lidar_msg.header.stamp),
                timeout=Duration(seconds=0.1)
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warning(f"TF lookup failed: {e}")
            return

        # === SAVE ===
        # self.lidar_camera_pairs.append((lidar_time, cam_matches[0][1], cam_matches[1][1], cam_matches[2][1]))
        self.save_lidar(lidar_msg, tf_l2g, lidar_time)
        self.save_images(cam_matches)

        self.get_logger().info(
            f"Saved best-aligned frame @ {lidar_time:.6f} (max Δt = {best[0]*1000:.1f} ms)"
        )

        # ---- RESET BUFFERS (consume everything up to this time) ----
        # if len(self.lidar_tstamps) > 500:
        #     # Save
        #     np.save(
        #         os.path.join(self.base_dir, f'lidar_tstamps_{lidar_time:.6f}.npy'),
        #         np.array(self.lidar_tstamps)
        #     )
        #     for nc in range(self.num_cameras):
        #         np.save(
        #             os.path.join(self.base_dir, f'cam{nc+1}_tstamps_{cam_matches[nc][1]:.6f}.npy'),
        #             np.array(self.cam_tstamps[nc])
        #         )
        #     np.save(
        #         os.path.join(self.base_dir, f'lidarcam_pair_tstamps_{lidar_time:.6f}.npy'),
        #         np.array(self.lidar_camera_pairs)
        #     )
            
        #     self.cam_tstamps = [[] for _ in range(self.num_cameras)]
        #     self.lidar_tstamps = []
        #     self.lidar_camera_pairs = []
            
        self.lidar_buffer = []
        for cam_id in range(self.num_cameras):
            self.cam_buffer[cam_id] = []

    def save_lidar(self, msg, transform_lidar2map, pc_time):
        lidar_pc_list = list(read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        ))
        lidar_pc_arr = np.array([list(p)[:4] for p in lidar_pc_list])

        np.save(
            os.path.join(self.lidar_pc_dir, f'lidar_pc_{pc_time:.6f}.npy'),
            lidar_pc_arr
        )

        tf_time = self.header_to_seconds(transform_lidar2map.header)
        np.save(
            os.path.join(self.trans_l2g_pc_dir, f'transform_lidar_to_map_{tf_time:.6f}.npy'),
            [
                transform_lidar2map.transform.translation.x,
                transform_lidar2map.transform.translation.y,
                transform_lidar2map.transform.translation.z,
                transform_lidar2map.transform.rotation.x,
                transform_lidar2map.transform.rotation.y,
                transform_lidar2map.transform.rotation.z,
                transform_lidar2map.transform.rotation.w,
            ]
        )
        if self.publish_extrinsics:
            for nc in range(self.num_cameras):
                np.save(os.path.join(self.trans_l2c_pc_dirs[nc], f'transform_lidar_to_cam{nc+1}_{pc_time:.6f}.npy'), [
                    self.trans_l2c[nc][0], # x 
                    self.trans_l2c[nc][1], # y 
                    self.trans_l2c[nc][2], # z
                    self.quat_l2c[nc][0], # rot.x, 
                    self.quat_l2c[nc][1], # rot.y, 
                    self.quat_l2c[nc][2], # rot.z, 
                    self.quat_l2c[nc][3], # rot.w,
                ])
        self.get_logger().info("Saved metric data")
        
    def save_images(self, closest_imgs):
        for cam_id, t, img in closest_imgs:
            cv2.imwrite(
                os.path.join(
                    self.cam_imgs_dirs[cam_id],
                    f'cam{cam_id+1}_img_{t:.6f}.jpg'
                ),
                img
            )
    
    def global_pc_callback(self, msg):
        global_pc_time = self.header_to_seconds(msg.header)
        global_pc_list = list(read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        global_pc_arr = np.array([list(p)[:4] for p in global_pc_list])
        np.save(os.path.join(self.global_pc_dir, f'global_pc_{global_pc_time:.6f}.npy'), global_pc_arr)   
    
    @staticmethod
    def header_to_seconds(header: Header) -> float:
        return header.stamp.sec + header.stamp.nanosec * 1e-9

def main(args=None):
    rclpy.init(args=args)
    node = SaveMetricDataMultiCam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
