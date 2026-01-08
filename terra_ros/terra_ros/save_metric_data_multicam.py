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
        super().__init__('save_metric_data_multicam')
        
        self.get_logger().info("Saving metric data")
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('save_folder', rclpy.Parameter.Type.STRING),
                ('lidar_topic', rclpy.Parameter.Type.STRING),
                ('num_cameras', rclpy.Parameter.Type.INTEGER),
                ('camera1_topic', rclpy.Parameter.Type.STRING),
                ('camera2_topic', rclpy.Parameter.Type.STRING),
                ('camera3_topic', rclpy.Parameter.Type.STRING),
                ('publish_extrinsics', rclpy.Parameter.Type.BOOL),
                ('lidar_to_camera1.translation', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lidar_to_camera1.rotation_quaternion', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lidar_to_camera1.parent_frame', rclpy.Parameter.Type.STRING),
                ('lidar_to_camera1.child_frame', rclpy.Parameter.Type.STRING),
                ('lidar_to_camera2.translation', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lidar_to_camera2.rotation_quaternion', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lidar_to_camera2.parent_frame', rclpy.Parameter.Type.STRING),
                ('lidar_to_camera2.child_frame', rclpy.Parameter.Type.STRING),
                ('lidar_to_camera3.translation', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lidar_to_camera3.rotation_quaternion', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('lidar_to_camera3.parent_frame', rclpy.Parameter.Type.STRING),
                ('lidar_to_camera3.child_frame', rclpy.Parameter.Type.STRING),
            ]
        ) # TODO: Make this self.declare_parameters flexible to variable number of cameras
        
        # Get parameters
        save_folder = self.get_parameter('save_folder').get_parameter_value().string_value
        lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.publish_extrinsics = self.get_parameter('publish_extrinsics').get_parameter_value().bool_value
        
        camera_topics = []
        translations = []
        rotations = []
        parent_frames = []
        child_frames = []
        for nc in range(self.num_cameras):
            camera_topics.append(
                self.get_parameter(f'camera{nc+1}_topic').get_parameter_value().string_value
            )
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
                
        self.cam_imgs = [None for _ in range(self.num_cameras)]
        self.cam_msgs = [None for _ in range(self.num_cameras)]
        self.lidar_pc = None
        self.global_pc = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        
        
    def cam_img_callback(self, msg, cam_id):
        img_data = np.array(msg.data)
        img = img_data.reshape(msg.height, msg.width, -1)  # Reshape if needed
        self.cam_imgs[cam_id] = img
        self.cam_msgs[cam_id] = msg
        cam_time = self.header_to_seconds(self.cam_msgs[cam_id].header)
        cv2.imwrite(
            os.path.join(self.cam_imgs_dirs[cam_id], 
                         f'cam{cam_id+1}_img_{cam_time}.jpg'), 
            self.cam_imgs[cam_id])
    
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
            np.save(os.path.join(self.lidar_pc_dir, f'lidar_pc_{pc_time}.npy'), lidar_pc_arr)

            # Save transformations
            np.save(os.path.join(self.trans_l2g_pc_dir, f'transform_lidar_to_map_{tf_time}.npy'), [
                transform_lidar2map.transform.translation.x, 
                transform_lidar2map.transform.translation.y, 
                transform_lidar2map.transform.translation.z,
                transform_lidar2map.transform.rotation.x, 
                transform_lidar2map.transform.rotation.y, 
                transform_lidar2map.transform.rotation.z, 
                transform_lidar2map.transform.rotation.w
            ])
            if self.publish_extrinsics:
                for nc in range(self.num_cameras):
                    np.save(os.path.join(self.trans_l2c_pc_dirs[nc], f'transform_lidar_to_cam{nc+1}_{pc_time}.npy'), [
                        self.trans_l2c[nc][0], # x 
                        self.trans_l2c[nc][1], # y 
                        self.trans_l2c[nc][2], # z
                        self.quat_l2c[nc][0], # rot.x, 
                        self.quat_l2c[nc][1], # rot.y, 
                        self.quat_l2c[nc][2], # rot.z, 
                        self.quat_l2c[nc][3], # rot.w,
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
        np.save(os.path.join(self.global_pc_dir, f'global_pc_{global_pc_time}.npy'), global_pc_arr)   
        
    def header_to_seconds(self, header: Header) -> float:
        return header.stamp.sec + header.stamp.nanosec * 1e-9

def main(args=None):
    rclpy.init(args=args)
    node = SaveMetricDataMultiCam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
