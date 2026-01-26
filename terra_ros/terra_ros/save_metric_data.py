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
class SaveMetricData(Node):
    def __init__(self):
        super().__init__('save_metric_data')
        
        self.get_logger().info("Saving metric data")
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('save_folder', rclpy.Parameter.Type.STRING),
                ('lidar_topic', rclpy.Parameter.Type.STRING),
                ('camera_topic', rclpy.Parameter.Type.STRING),
                ('publish_extrinsics', rclpy.Parameter.Type.BOOL),
                ('lidar_to_camera.translation', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lidar_to_camera.rotation_quaternion', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lidar_to_camera.parent_frame', rclpy.Parameter.Type.STRING),
                ('lidar_to_camera.child_frame', rclpy.Parameter.Type.STRING),
            ]
        )
        # Get parameters
        save_folder = self.get_parameter('save_folder').get_parameter_value().string_value
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.publish_extrinsics = self.get_parameter('publish_extrinsics').get_parameter_value().bool_value
        translation = self.get_parameter('lidar_to_camera.translation').get_parameter_value().double_array_value
        rotation = self.get_parameter('lidar_to_camera.rotation_quaternion').get_parameter_value().double_array_value
        parent_frame = self.get_parameter('lidar_to_camera.parent_frame').get_parameter_value().string_value
        child_frame = self.get_parameter('lidar_to_camera.child_frame').get_parameter_value().string_value
        # Log for debugging
        self.get_logger().info(f'save_folder: {save_folder}')
        self.get_logger().info(f'lidar_topic: {lidar_topic}')
        self.get_logger().info(f'camera_topic: {camera_topic}')
        self.get_logger().info(f'publish_extrinsics: {self.publish_extrinsics}')
        self.get_logger().info(f'translation: {translation}')
        self.get_logger().info(f'rotation_quaternion: {rotation}')
        self.get_logger().info(f'{parent_frame} -> {child_frame}')        
        
        
        self.local_cam_img = None
        self.local_cam_msg = None
        self.lidar_pc = None
        self.global_pc = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.local_cam_img_sub = self.create_subscription(
            Image,
            camera_topic,
            self.local_cam_img_callback,
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
        
        if self.publish_extrinsics:
            trans_C2L_in_C = np.array(translation)  # [x, y, z]
            rot = R.from_quat(np.array(rotation))
            self.quat_l2c = rot.as_quat()  # [qx, qy, qz, qw]
            self.trans_l2c = trans_C2L_in_C # [x,y,z]

        self.base_dir = save_folder
        os.makedirs(self.base_dir, exist_ok=True)
        self.lidar_pc_dir = os.path.join(self.base_dir, 'lidar_pc')
        os.makedirs(self.lidar_pc_dir, exist_ok=True)
        self.global_pc_dir = os.path.join(self.base_dir, 'global_pc')
        os.makedirs(self.global_pc_dir, exist_ok=True)
        self.local_imgs_dir = os.path.join(self.base_dir, 'camera1_images')
        os.makedirs(self.local_imgs_dir, exist_ok=True)
        self.trans_l2g_pc_dir = os.path.join(self.base_dir, 'transformations_lidar2global')
        os.makedirs(self.trans_l2g_pc_dir, exist_ok=True)
        self.trans_l2c_pc_dir = os.path.join(self.base_dir, 'transformations_lidar2cam1')
        os.makedirs(self.trans_l2c_pc_dir, exist_ok=True)

    def local_cam_img_callback(self, msg):
        img_data = np.array(msg.data)
        img = img_data.reshape(msg.height, msg.width, -1)  # Reshape if needed
        self.local_cam_img = img
        self.local_cam_msg = msg
        cam_time = self.header_to_seconds(self.local_cam_msg.header)
        cv2.imwrite(os.path.join(self.local_imgs_dir, f'cam1_img_{cam_time:.6f}.jpg'), self.local_cam_img)
    
    def lidar_pc_callback(self, msg):
        try:
            self.lidar_pc = msg
            pc_time = self.header_to_seconds(self.lidar_pc.header)
            lidar_pc_list = list(read_points(self.lidar_pc, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            local_pts = []
            for p_idx in range(len(lidar_pc_list)):
                local_pts.append(list(lidar_pc_list[p_idx])[:4])
            lidar_pc_arr = np.array(local_pts) # (num_pts, 4)
            # transform_lidar2map = self.tf_buffer.lookup_transform('map', msg.header.frame_id, Time.from_msg(msg.header.stamp))
            transform_lidar2map = self.tf_buffer.lookup_transform('map', msg.header.frame_id, Time())
            tf_time = self.header_to_seconds(transform_lidar2map.header)
            
            # Save lidar scan
            np.save(os.path.join(self.lidar_pc_dir, f'lidar_pc_{pc_time:.6f}.npy'), lidar_pc_arr)

            # Save transformations
            np.save(os.path.join(self.trans_l2g_pc_dir, f'transform_lidar_to_map_{tf_time:.6f}.npy'), [
                transform_lidar2map.transform.translation.x, 
                transform_lidar2map.transform.translation.y, 
                transform_lidar2map.transform.translation.z,
                transform_lidar2map.transform.rotation.x, 
                transform_lidar2map.transform.rotation.y, 
                transform_lidar2map.transform.rotation.z, 
                transform_lidar2map.transform.rotation.w
            ])
            if self.publish_extrinsics:
                np.save(os.path.join(self.trans_l2c_pc_dir, f'transform_lidar_to_cam1_{pc_time:.6f}.npy'), [
                    self.trans_l2c[0], # x 
                    self.trans_l2c[1], # y 
                    self.trans_l2c[2], # z
                    self.quat_l2c[0], # rot.x, 
                    self.quat_l2c[1], # rot.y, 
                    self.quat_l2c[2], # rot.z, 
                    self.quat_l2c[3], # rot.w,
                ])
            self.get_logger().info("Saved lidar data")
        except Exception as e:
            print("ERROR with exception:",e)
            self.get_logger().warning(f"Failed to save lidar data b/c of {e}")
        
    def global_pc_callback(self, msg):
        self.global_pc = msg
        global_pc_time = self.header_to_seconds(self.global_pc.header)
        global_pc_list = list(read_points(self.global_pc, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        global_pts = []
        for p_idx in range(len(global_pc_list)):
            global_pts.append(list(global_pc_list[p_idx])[:4])
        global_pc_arr = np.array(global_pts) # (num_pts, 4)
        np.save(os.path.join(self.global_pc_dir, f'global_pc_{global_pc_time:.6f}.npy'), global_pc_arr)   
        
    def header_to_seconds(self, header: Header) -> float:
        return header.stamp.sec + header.stamp.nanosec * 1e-9

def main(args=None):
    rclpy.init(args=args)
    node = SaveMetricData()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
