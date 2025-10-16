import os
import yaml

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Path to the launch file in the 'ouster-ros' package
    lidar_launch_file = os.path.join(
        get_package_share_directory('ouster_ros'),  # Replace with the exact package name
        'launch',
        'driver.launch.py'  # Replace with the correct launch file name
    )
    ouster_params_file = os.path.join(
        get_package_share_directory('ouster_ros'),  # Replace with your package name
        'config',
        'driver_params.yaml'  # Make sure this file exists and contains the parameters
    )
    
    oakd_launch_file = os.path.join(
        get_package_share_directory('depthai_ros_driver'),  # Replace with the exact package name
        'launch',
        'rgbd_pcl.launch.py'#'camera.launch.py'
    )
    
    liosam_launch_file = os.path.join(
        get_package_share_directory('lio_sam'),  # Replace with the exact package name
        'launch',
        'run.launch.py'  # Replace with the correct launch file name
    )
    liosam_params_file = os.path.join(
        get_package_share_directory('lio_sam'),  # Replace with your package name
        'config',
        'params.yaml'  # Make sure this file exists and contains the parameters
    )
    
    transforms_file = os.path.join(
        get_package_share_directory('oasis2'),  # Replace with your package name
        'config',
        'extrinsic_transforms.yaml'
    )
    with open(transforms_file, 'r') as f:
        transforms = yaml.safe_load(f)
    lidar_to_camera = transforms['camera_to_lidar']
    
    
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lidar_launch_file),
            launch_arguments={'params_file': ouster_params_file}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(oakd_launch_file),
        ),

        # Publish static extrinsic transformations
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                '--x', str(lidar_to_camera['translation'][0]),
                '--y', str(lidar_to_camera['translation'][1]),
                '--z', str(lidar_to_camera['translation'][2]),
                '--qx', str(lidar_to_camera['rotation_quaternion'][0]),
                '--qy', str(lidar_to_camera['rotation_quaternion'][1]),
                '--qz', str(lidar_to_camera['rotation_quaternion'][2]),
                '--qw', str(lidar_to_camera['rotation_quaternion'][3]),
                '--frame-id', lidar_to_camera['parent_frame'],
                '--child-frame-id', lidar_to_camera['child_frame']
            ],
            output='screen',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(liosam_launch_file),
            launch_arguments={'params_file': liosam_params_file}.items(),
        ),
    ])