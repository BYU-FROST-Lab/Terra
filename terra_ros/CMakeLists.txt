cmake_minimum_required(VERSION 3.8)
project(terra_ros)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Install the Python executable scripts
install(
  DIRECTORY terra_ros/
  DESTINATION lib/${PROJECT_NAME}
)
# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ament package macros
ament_export_dependencies(rosidl_default_runtime rclpy std_msgs sensor_msgs)
ament_package()
