import os
import yaml

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def setup_nodes(context, *args, **kwargs):
    # Resolve the YAML path from LaunchConfiguration at runtime
    metric_map_params_file = LaunchConfiguration('params_file').perform(context)

    # Load YAML
    with open(metric_map_params_file, 'r') as f:
        raw_params = yaml.safe_load(f)

    params = raw_params.get('save_metric_data_multicam_fix_rate', {}).get('ros__parameters', {})
    publish_extrinsics = params['publish_extrinsics']
    num_cameras = int(params['num_cameras'])
    lidar_to_cameras = [params['lidar_to_camera'+str(i+1)] for i in range(num_cameras)]
    rosbag_path = params['rosbag_path']

    # LIO-SAM paths
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

    actions = []

    # Include LIO-SAM
    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(liosam_launch_file),
            launch_arguments={'params_file': liosam_params_file}.items(),
        )
    )

    # Play bag
    actions.append(
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        'ros2', 'bag', 'play', rosbag_path,
                        '-r', '0.5', '--clock',# '--read-ahead-queue-size', '100000'
                    ],
                    output='screen'
                )
            ]
        )
    )

    # Save metric data node
    actions.append(
        Node(
            package='terra_ros',
            executable='save_metric_data_multicam_fix_rate.py',
            name='save_metric_data_multicam_fix_rate',
            parameters=[
                metric_map_params_file, 
                {'use_sim_time': True}
            ],
            output='screen'
        )
    )

    # Static transforms
    if publish_extrinsics:
        for i in range(num_cameras):
            actions.append(Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                arguments=[
                    '--x', str(lidar_to_cameras[i]['translation'][0]),
                    '--y', str(lidar_to_cameras[i]['translation'][1]),
                    '--z', str(lidar_to_cameras[i]['translation'][2]),
                    '--qx', str(lidar_to_cameras[i]['rotation_quaternion'][0]),
                    '--qy', str(lidar_to_cameras[i]['rotation_quaternion'][1]),
                    '--qz', str(lidar_to_cameras[i]['rotation_quaternion'][2]),
                    '--qw', str(lidar_to_cameras[i]['rotation_quaternion'][3]),
                    '--frame-id', lidar_to_cameras[i]['parent_frame'],
                    '--child-frame-id', lidar_to_cameras[i]['child_frame']
                ],
                output='screen'
            ))

    return actions


def generate_launch_description():
    # Declare launch argument
    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('terra_ros'),
            'config', 'multicam', 'params_rate.yaml'
        ),
        description='Full path to the params_rate.yaml file'
    )

    # Use OpaqueFunction to access the YAML at runtime
    return LaunchDescription([
        declare_params_file_arg,
        OpaqueFunction(function=setup_nodes)
    ])
