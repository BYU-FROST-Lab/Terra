from argparse import ArgumentParser
import yaml
import numpy as np
import matplotlib.pyplot as plt

from terra_utils import load_terra
import numpy as np
import pickle as pkl


'''
MIT License
Copyright (c) 2019 Michail Kalaitzakis
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

class GPS_utils:
	'''
		Contains the algorithms to convert a gps signal (longitude, latitude, height)
		to a local cartesian ENU system and vice versa
		
		Use setENUorigin(lat, lon, height) to set the local ENU coordinate system origin
		Use geo2enu(lat, lon, height) to get the position in the local ENU system
		Use enu2geo(x_enu, y_enu, z_enu) to get the latitude, longitude and height
	'''
	
	def __init__(self):
		# Geodetic System WGS 84 axes
		self.a  = 6378137.0
		self.b  = 6356752.314245
		self.a2 = self.a * self.a
		self.b2 = self.b * self.b
		self.e2 = 1.0 - (self.b2 / self.a2)
		self.e  = self.e2 / (1.0 - self.e2)
		
		# Local ENU Origin
		self.latZero = None
		self.lonZero = None
		self.hgtZero = None
		self.xZero = None
		self.yZero = None
		self.zZero = None
		self.R = np.asmatrix(np.eye(3))

	def setENUorigin(self, lat, lon, height):
		# Save origin lat, lon, height
		self.latZero = lat
		self.lonZero = lon
		self.hgtZero = height
		
		# Get origin ECEF X,Y,Z
		origin = self.geo2ecef(self.latZero, self.lonZero, self.hgtZero)		
		self.xZero = origin.item(0)
		self.yZero = origin.item(1)
		self.zZero = origin.item(2)
		self.oZero = np.array([[self.xZero], [self.yZero], [self.zZero]])
		
		# Build rotation matrix
		phi = np.deg2rad(self.latZero)
		lmd = np.deg2rad(self.lonZero)
		
		cPhi = np.cos(phi)
		cLmd = np.cos(lmd)
		sPhi = np.sin(phi)
		sLmd = np.sin(lmd)
		
		self.R[0, 0] = -sLmd
		self.R[0, 1] =  cLmd
		self.R[0, 2] =  0.0
		self.R[1, 0] = -sPhi * cLmd
		self.R[1, 1] = -sPhi * sLmd
		self.R[1, 2] =  cPhi
		self.R[2, 0] =  cPhi * cLmd
		self.R[2, 1] =  cPhi * sLmd
		self.R[2, 2] =  sPhi
	
	def geo2ecef(self, lat, lon, height):
		phi = np.deg2rad(lat)
		lmd = np.deg2rad(lon)
		
		cPhi = np.cos(phi)
		cLmd = np.cos(lmd)
		sPhi = np.sin(phi)
		sLmd = np.sin(lmd)
		
		N = self.a / np.sqrt(1.0 - self.e2 * sPhi * sPhi)
		
		x = (N + height) * cPhi * cLmd
		y = (N + height) * cPhi * sLmd
		z = ((self.b2 / self.a2) * N + height) * sPhi
		
		return np.array([[x], [y], [z]])
	
	def ecef2enu(self, x, y, z):
		ecef = np.array([[x], [y], [z]])
		
		return self.R * (ecef - self.oZero)
	
	def geo2enu(self, lat, lon, height):
		ecef = self.geo2ecef(lat, lon, height)
		
		return self.ecef2enu(ecef.item(0), ecef.item(1), ecef.item(2)), (1, 3)
	
	def ecef2geo(self, x, y, z):
		p = np.sqrt(x*x + y*y)
		q = np.arctan2(self.a * z, self.b * p)
		
		sq = np.sin(q)
		cq = np.cos(q)
		
		sq3 = sq * sq * sq
		cq3 = cq * cq * cq
		
		phi = np.arctan2(z + self.e * self.b * sq3, p - self.e2 * self.a * cq3)
		lmd = np.arctan2(y, x)
		v = self.a / np.sqrt(1.0 - self.e2 * np.sin(phi) * np.sin(phi))

		lat = np.rad2deg(phi)
		lon = np.rad2deg(lmd)		
		h = (p / np.cos(phi)) - v
		
		return np.array([[lat], [lon], [h]])
		
	def enu2ecef(self, x, y, z):
		lmd = np.deg2rad(self.latZero)
		phi = np.deg2rad(self.lonZero)
		
		cPhi = np.cos(phi)
		cLmd = np.cos(lmd)
		sPhi = np.sin(phi)
		sLmd = np.sin(lmd)
		
		N = self.a / np.sqrt(1.0 - self.e2 * sLmd * sLmd)
		
		x0 = (self.hgtZero + N) * cLmd * cPhi
		y0 = (self.hgtZero + N) * cLmd * sPhi
		z0 = (self.hgtZero + (1.0 - self.e2) * N) * sLmd
		
		xd = -sPhi * x - cPhi * sLmd * y + cLmd * cPhi * z
		yd =  cPhi * x - sPhi * sLmd * y + cLmd * sPhi * z
		zd =  cLmd * y + sLmd * z
		
		return np.array([[x0+xd], [y0+yd], [z0+zd]])
	
	def enu2geo(self, x, y, z):
		ecef = self.enu2ecef(x, y, z)
		
		return self.ecef2geo(ecef.item(0), ecef.item(1), ecef.item(2)), (1, 3)


class Node_With_GPS:
    def __init__(self, node_id, GPS, pc_xy=None, ENU=None):
        self.node_id = node_id
        self.GPS = GPS
        self.pc_xy = pc_xy  # Placeholder for point cloud coordinates of the node
        self.ENU = ENU  # Placeholder for ENU coordinates of the node

if __name__ == '__main__':
    parser = ArgumentParser(description="Adding GPS coordinates to place nodes")
    parser.add_argument(
        '--params',
        type=str,
        help="/path/to/params.yaml file of region monitoring tasks"
    )
    args = parser.parse_args()
    
    with open(args.params, 'rb') as f:
        params = yaml.safe_load(f)


    terra = load_terra(params['terra'])
    nodes_with_gps = params['nodes_with_GPS']  # List of nodes with GPS coordinates from the config file

	# Initialize GPS utilities and set the ENU origin based on the first node with GPS coordinates
    gps_utils = GPS_utils()
    node_1 = Node_With_GPS(node_id=nodes_with_gps[0]['node_id'], GPS=nodes_with_gps[0]['GPS'], pc_xy=terra.terra_3dsg.nodes[nodes_with_gps[0]['node_id']]['pos'], ENU=np.array([0, 0, 0]))
    gps_utils.setENUorigin(node_1.GPS[0], node_1.GPS[1], 0)

    node_2 = Node_With_GPS(node_id=nodes_with_gps[1]['node_id'], GPS=nodes_with_gps[1]['GPS'], pc_xy=terra.terra_3dsg.nodes[nodes_with_gps[1]['node_id']]['pos'])
    node_2.ENU = np.ravel(gps_utils.geo2enu(node_2.GPS[0], node_2.GPS[1], 0)[0])  # Get ENU coordinates for the second node with GPS coordinates

    print(f"Node {node_1.node_id} - GPS: {node_1.GPS}, Point Cloud XY: {node_1.pc_xy}, ENU: {node_1.ENU}")
    print(f"Node {node_2.node_id} - GPS: {node_2.GPS}, Point Cloud XY: {node_2.pc_xy}, ENU: {node_2.ENU}")

    # Create transformation from point cloud coordinates to ENU coordinates using the two nodes with GPS coordinates
    v_pc = np.array(node_2.pc_xy) - np.array(node_1.pc_xy)  # Vector in point cloud space
    v_enu = node_2.ENU - node_1.ENU
    # v_enu = v_enu[0:2]  # Corresponding vector in ENU space

    print("v_pc:", v_pc)
    print("V_pc shape:", v_pc.shape)
    print("v_enu:", v_enu)
    print("v_enu shape:", v_enu.shape)
    
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

    print("Rotation matrix:\n", R)

    # Compute scale
    # scale = np.linalg.norm(v_enu) / np.linalg.norm(v_pc)
    scale = 1.0  # For now, just set scale to 1

    print("Scale:\n", scale)

    # Compute translation
    t = node_1.ENU[0:2] - scale * R @ node_1.pc_xy

    # Function to transform any point
    def pc_to_enu(point):
        return scale * R @ point + t


    node_gps_dict = {}  # Dictionary to store {node_id: GPS_coordinates pairs}
    node_enu_dict = {}  # Dictionary to store {node_id: ENU_coordinates pairs}
    node_gps_dict[node_1.node_id] = node_1.GPS
    node_gps_dict[node_2.node_id] = node_2.GPS
    node_enu_dict[node_1.node_id] = node_1.ENU
    node_enu_dict[node_2.node_id] = node_2.ENU


    place_nodes = [
        n for n, d in terra.terra_3dsg.nodes(data=True)
        if d["level"] == 1
    ]
    print(f"Total number of place nodes: {len(place_nodes)}")
    for node in place_nodes:
        if node in node_gps_dict:
           pass  # GPS coordinates already added for this node
        else:
            #Compute GPS coordinaters for this node based on its location and the GPS coordinates of the specified nodes
            pc_xy = terra.terra_3dsg.nodes[node]['pos']
            enu = pc_to_enu(pc_xy)
            gps = np.ravel(gps_utils.enu2geo(enu[0], enu[1], 0)[0])[0:2]
            node_gps_dict[node] = gps.flatten().tolist()  # Store GPS coordinates in the dictionary
            node_enu_dict[node] = enu.flatten().tolist()  # Store ENU coordinates in the dictionary
			
    # Save the dictionaries and transformation to a pickle file
    name = "provo_river"
    with open(f'node_gps_dict_{name}.pkl', 'wb') as f:
        pkl.dump(node_gps_dict, f)
    with open(f'node_enu_dict_{name}.pkl', 'wb') as f:
        pkl.dump(node_enu_dict, f)
    with open(f'pc_to_enu_transformation_{name}.pkl', 'wb') as f:
        pkl.dump({'rotation': R, 'scale': scale, 'translation': t}, f)
        

    # Print out all the GPS coordinates for each node
    for node_id, gps in node_gps_dict.items():
        print(gps[0], ", ", gps[1])



    #Plot two plots, pc space on left, ENU on left
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    pc_xs = []
    pc_ys = []
    for node_id, gps in node_gps_dict.items():
        pc_xy = terra.terra_3dsg.nodes[node_id]['pos']
        pc_xs.append(pc_xy[0])
        pc_ys.append(pc_xy[1])
    
    plt.scatter(pc_xs, pc_ys, s=5)
    plt.title("Place Nodes with GPS Coordinates in Point Cloud Space")
    plt.xlabel("Point Cloud X")
    plt.ylabel("Point Cloud Y")
    plt.grid()


    # Plot the nodes with GPS coordinates in the ENU space
    plt.subplot(1, 2, 2)
    enu_x = []
    enu_y = []
    for node_id, gps in node_gps_dict.items():
        enu = node_enu_dict[node_id]
        enu_x.append(enu[0])
        enu_y.append(enu[1])
    plt.scatter(enu_x, enu_y, s=5)
    plt.title('Place Nodes with GPS Coordinates in ENU Space')
    plt.xlabel('ENU X')
    plt.ylabel('ENU Y')
    plt.grid()
    plt.show() 
            