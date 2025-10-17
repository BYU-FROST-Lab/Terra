# Terra
This repository contains the code for *Terra: Hierarchical Terrain-Aware 3D Scene Graph for Task-Agnostic Outdoor Mapping*

![Terra](./assets/method_fig.png)


# Table of Contents
* [Paper](#Paper)
* [Setup](#Setup)
* [Datasets](#Datasets)
* [Building Terra](#building-terra)
* [Task Execution with Terra](#task-execution-with-terra)

# Paper

*In Review*

# Setup

### Requirements
* [Docker](https://docs.docker.com/engine/install/ubuntu/)
* NVIDIA GPU

Our repository is designed for ROS 2 Humble. To handle dependency issues, our repository is built in Docker using the following container.

### Using Docker Container

First pull our Terra docker container
```bash
docker pull frostlab/ros2_terra:latest
xhost +local:docker
```

Start the container in your terminal
```bash
docker run --rm -it --gpus all --net host -v /tmp/.X11-unix:/tmp/.X11-unix -v <path/to/terra_repo>:/docker_ros2_ws/src/terra -v <ros2_bags_folder>:/<ros2_bags_folder> -e DISPLAY=$DISPLAY frostlab/ros2_terra
```
Build and source the ros2 repository
```
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

# Datasets

### Simulation: Business Campus

### Real-World: South Campus

### Custom Datasets

<details open>

<summary><b>Preparing Data</b></summary>

If you have a ROS 2 bag of your Ouster OS1-128 LiDAR and RGB Camera data, then do the following to run LIO-SAM:

- Edit the `params.yaml` file in the `LIO-SAM/config` folder to match your LiDAR and IMU ros topics as well as the extrinsic transformation between the two.
- Update the `params.yaml` ros parameters in `terra_ros/config` for your dataset and rosbag
- Now to build the metric point cloud map with LIO-SAM and save the data into our folder structure, run
```bash
ros2 launch terra_ros build_metric_map.launch.py
```

</details>

Our dataset folder structure:
```
my_dataset
├── global_pc
|   ├── global_pc_{timestamp}.npy
|   ├── ...
├── local_images
|   ├── local_cam_img_{timestamp}.jpg
|   ├── ...
├── local_pc
|   ├── local_pc_{timestamp}.npy
|   ├── ...
├── transformations_lidar2cam
|   ├── transform_lidar_to_cam_{timestamp}.npy
|   ├── ...
├── transformations_lidar2global
|   ├── transform_lidar_to_map_{timestamp}.npy
|   └── ...
```

# Building Terra

<details open>

<summary><b>Building Metric-Semantic Map (MS Map)</b></summary>

</details>


<details open>

<summary><b>Building Region and Place Layers from MS Map</b></summary>


</details>

# Task-Execution with Terra


<details open>

<summary><b>Object Retrieval Tasks</b></summary>


</details>


<details open>

<summary><b>Region Monitoring Tasks</b></summary>


</details>


<details open>

<summary><b>Terrain-Aware Path Planning for Object Retrieval</b></summary>


</details>