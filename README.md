# Terra
This repository contains the code for *Terra: Hierarchical Terrain-Aware 3D Scene Graph for Task-Agnostic Outdoor Mapping*

![Terra](./assets/method_fig.png)


# Table of Contents
- [Terra](#terra)
- [Table of Contents](#table-of-contents)
- [Paper](#paper)
- [Setup](#setup)
    - [Requirements](#requirements)
    - [Using Docker Container](#using-docker-container)
- [Datasets](#datasets)
- [Building Terra](#building-terra)
- [Task-Execution with Terra](#task-execution-with-terra)

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

<summary><b>Dataset folder structure</b></summary>

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

</details>

<details open>

<summary><b>Simulation: Business Campus Dataset</b></summary>

- Download the dataset [here](https://gofile.me/7dj2d/y7R2ETLpK) (~116 GB)
- Put the rosbag in the volumed `ros2_bags_folder` so you can access it in the docker container 
- Copy the contents of the `config/sim/liosam_params.yaml` file to replace the contents in the `params.yaml` file in the `LIO-SAM/config` folder
- Now to build the metric point cloud map with LIO-SAM and save the data into our folder structure, run
```bash
ros2 launch terra_ros build_metric_map_sim.launch.py
```
</details>


<details open>

<summary><b>Real-World: South Campus Dataset</b></summary>

- Download the dataset [here](https://gofile.me/7dj2d/JvUpRm12E) (~263 GB)
- Put the rosbag in the volumed `ros2_bags_folder` so you can access it in the docker container 
- Copy the contents of the `config/south_campus/liosam_params.yaml` file to replace the contents in the `params.yaml` file in the `LIO-SAM/config` folder
- Now to build the metric point cloud map with LIO-SAM and save the data into our folder structure, run
```bash
ros2 launch terra_ros build_metric_map_sim.launch.py
```
</details>


<details open>

<summary><b>Custom Datasets</b></summary>

If you have a ROS 2 bag of your Ouster OS1-128 LiDAR and RGB Camera data, then do the following to run LIO-SAM:

- Edit the `params.yaml` file in the `LIO-SAM/config` folder to match your LiDAR and IMU ros topics as well as the extrinsic transformation between the two.
- Update the `params.yaml` ros parameters in `terra_ros/config` for your dataset and rosbag
- Now to build the metric point cloud map with LIO-SAM and save the data into our folder structure, run
```bash
ros2 launch terra_ros build_metric_map.launch.py
```
</details>

<details open>


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