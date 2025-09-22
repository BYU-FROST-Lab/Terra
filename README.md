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
* Docker installed

Our repository is designed for ROS 2 Humble. To handle dependency issues, our repository is built in Docker using the following container.

### Using Docker Container

First pull our Terra docker container with:
```bash
docker pull frostlab/ros2_terra:latest
xhost +local:docker
```

To start the container in your terminal, run
```bash
docker run --rm -it --gpus all --net host -v /tmp/.X11-unix:/tmp/.X11-unix -v <path/to/terra_repo>:/docker_ros2_ws/src/terra -e DISPLAY=$DISPLAY frostlab/ros2_terra
```

# Datasets

ROS 2 dataset folder structure example:
```
south_campus
├── south_campus.db3
└── metadata.yaml
```

### Simulation: Business Campus

### Real-World: South Campus

### Custom Datasets

<details open>

<summary><b>Preparing Data</b></summary>

If you have a ROS 2 bag of your Ouster OS1-128 LiDAR and RGB Camera data, then do the following to run LIO-SAM:
```bash

```

</details>


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

<summary><b>Path Planning from Object Retrieval</b></summary>


</details>