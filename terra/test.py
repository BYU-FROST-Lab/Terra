import numpy as np


class Node_With_GPS:
    def __init__(self, node_id, GPS, pc_xy=None, ENU=None):
        self.node_id = node_id
        self.GPS = GPS
        self.pc_xy = pc_xy  # Placeholder for point cloud coordinates of the node
        self.ENU = ENU  # Placeholder for ENU coordinates of the node

def test_transformation(node_1, node_2, point_3_pc, expected_point_3_enu):

    # Create transformation from point cloud coordinates to ENU coordinates using the two nodes with GPS coordinates
    v_pc = np.array(node_2.pc_xy) - np.array(node_1.pc_xy)  # Vector in point cloud space
    v_enu = (node_2.ENU - node_1.ENU)[0:2]  # Corresponding vector in ENU space

    print("v_pc:", v_pc)
    print("v_enu:", v_enu)


    # Compute rotation
    theta = (
        np.arctan2(v_enu[1], v_enu[0])
        - np.arctan2(v_pc[1], v_pc[0])
    )

    print("Theta (radians):", theta)
    print("Theta (degrees):", np.degrees(theta))

    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Compute scale
    scale = np.linalg.norm(v_enu) / np.linalg.norm(v_pc)

    # Compute translation
    t = node_1.ENU[0:2] - scale * R @ node_1.pc_xy

    # Function to transform any point
    def pc_to_enu(point):
        return scale * R @ point + t


    print("Rotation matrix:\n", R)
    print("Scale:\n", scale)
    print("Translation:\n", t)

    point_3_enu = pc_to_enu(point_3_pc)
    print("Point 3 in ENU space:", point_3_enu)
    print("Expected Point 3 in ENU space:", expected_point_3_enu)



# Test that I'm computing the transformation from point cloud coordinates to ENU coordinates correctly
# node_1 = Node_With_GPS(node_id=0, GPS=[40.12345, -111.12345], pc_xy=[0, 0], ENU=np.array([0, 0, 0]))
# node_2 = Node_With_GPS(node_id=1, GPS=[40.54321, -111.54321], pc_xy=[10, 0], ENU=np.array([10, 0, 0]))

# Test rotation of 90 degrees
print("Test 1: Rotation of 90 degrees")
node_1 = Node_With_GPS(node_id=0, GPS=[40.12345, -111.12345], pc_xy=[0, 0], ENU=np.array([0, 0, 0]))
node_2 = Node_With_GPS(node_id=1, GPS=[40.54321, -111.54321], pc_xy=[10, 0], ENU=np.array([0, 10, 0]))
point_3_pc = np.array([5, 0])  # Midpoint in point cloud space
expected_point_3_enu = np.array([0, 5])  # Expected position in ENU space
test_transformation(node_1, node_2, point_3_pc, expected_point_3_enu)

# Test translation only (no rotation)
print("\n\nTesting translation only (no rotation):")
node_1 = Node_With_GPS(node_id=0, GPS=[40.12345, -111.12345], pc_xy=[0, 0], ENU=np.array([10, 20, 0]))
node_2 = Node_With_GPS(node_id=1, GPS=[40.54321, -111.54321], pc_xy=[10, 0], ENU=np.array([20, 20, 0]))
point_3_pc = np.array([5, 0])  # Midpoint in point cloud space
expected_point_3_enu = np.array([15, 20])  # Expected position in ENU space
test_transformation(node_1, node_2, point_3_pc, expected_point_3_enu)

# Test scaling only (no rotation, no translation)
print("\n\nTesting scaling only (no rotation, no translation):")
node_1 = Node_With_GPS(node_id=0, GPS=[40.12345, -111.12345], pc_xy=[0, 0], ENU=np.array([0, 0, 0]))
node_2 = Node_With_GPS(node_id=1, GPS=[40.54321, -111.54321], pc_xy=[10, 0], ENU=np.array([20, 0, 0]))  # Scale of 2
point_3_pc = np.array([5, 0])  # Midpoint in point cloud space
expected_point_3_enu = np.array([10, 0])  # Expected position in ENU space
test_transformation(node_1, node_2, point_3_pc, expected_point_3_enu)




