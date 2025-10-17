import os
import yaml

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    metric_map_params_file = os.path.join(
        get_package_share_directory('terra_ros'),
        'config', 'params.yaml'
    )
    # params_declare = DeclareLaunchArgument(
    #     'params_file',
    #     default_value=metric_map_params_file)
    # params_file = LaunchConfiguration('params_file')
    with open(metric_map_params_file, 'r') as f:
        raw_params = yaml.safe_load(f)
    params = raw_params.get('save_metric_data', {}).get('ros__parameters', {})
    publish_extrinsics = params['publish_extrinsics']
    lidar_to_camera = params['lidar_to_camera']
    rosbag_path = params['rosbag_path']
    
    # lidar_launch_file = os.path.join(
    #     get_package_share_directory('ouster_ros'),
    #     'launch',
    #     'driver.launch.py'
    # )
    # ouster_params_file = os.path.join(
    #     get_package_share_directory('ouster_ros'),
    #     'config',
    #     'driver_params.yaml'
    # )
    
    # oakd_launch_file = os.path.join(
    #     get_package_share_directory('depthai_ros_driver'),
    #     'launch',
    #     'camera.launch.py'
    # )
    
    liosam_launch_file = os.path.join(
        get_package_share_directory('lio_sam'),
        'launch',
        'run.launch.py'
    )
    liosam_params_file = os.path.join(
        get_package_share_directory('lio_sam'),
        'config',
        'params.yaml'
    )  
    
    actions = [
        # params_declare,
        
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(lidar_launch_file),
        #     launch_arguments={'params_file': ouster_params_file}.items(),
        # ),
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(oakd_launch_file),
        # ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(liosam_launch_file),
            launch_arguments={'params_file': liosam_params_file}.items(),
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', rosbag_path],#, '-r 0.5'],
            output='screen'
        ),
        Node(
            package='terra_ros',
            executable='save_metric_data.py',
            name='save_metric_data',
            parameters=[metric_map_params_file],
            output='screen',
        ),
    ]
    
    if publish_extrinsics:
        # Publish static extrinsic transformations
        actions.append(Node(
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
        ))
    
    return LaunchDescription(actions)