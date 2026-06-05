from argparse import ArgumentParser
import yaml
import torch
import clip

from terra_utils import load_terra
import numpy as np
from matplotlib.path import Path
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


class RegionGPSVisualizer:
    def __init__(self, terra, node_gps_dict, node_enu_dict, R, scale, t, node_origin_id=None):
        self.terra = terra
        self.node_gps_dict = node_gps_dict
        self.node_enu_dict = node_enu_dict
        self.transformation = transformation
        self.R = R
        self.scale = scale
        self.t = t
        self.GPSUtils = GPS_utils()
        if node_origin_id is not None:
            origin_gps = self.node_gps_dict[node_origin_id]
            self.GPSUtils.setENUorigin(origin_gps[0], origin_gps[1], 0.0)  # Assuming height is 0 for the origin node

    def get_nodes_in_rectangle_from_refs(self, gps_coords):
        """
        gps_coords: list of 2 or 4 GPS coordinates corresponding to the corners of the rectangle in clockwise order starting from top left
        returns: list of node ids that are within the rectangle defined by the gps coordinates
        """
        # Convert GPS coordinates to ENU coordinates
        enu_coords = []
        for gps in gps_coords:
            enu, _ = self.GPSUtils.geo2enu(gps[0], gps[1], 0.0)  # Assuming height is 0 for the rectangle corners
            enu_coords.append(enu.flatten().tolist())
        enu_coords = np.array(enu_coords)

        # Get the min and max ENU coordinates to define the rectangle in ENU space
        min_x, min_y = np.min(enu_coords, axis=0)
        max_x, max_y = np.max(enu_coords, axis=0)

        # Find nodes within the ENU rectangle
        nodes_in_rectangle = []
        for node_id, enu in self.node_enu_dict.items():
            if (min_x <= enu[0] <= max_x) and (min_y <= enu[1] <= max_y):
                nodes_in_rectangle.append(node_id)

        return nodes_in_rectangle
    
    def get_nodes_in_polygon_from_refs(self, gps_coords):
        """
        gps_coords: list of GPS coordinates corresponding to the corners of the polygon in clockwise order starting from top left
        returns list of node ids that are within the polygon defined by the gps coordinates
        """
        # Convert GPS coordinates to ENU coordinates
        enu_coords = []
        for gps in gps_coords:
            enu = np.ravel(self.GPSUtils.geo2enu(gps[0], gps[1], 0.0)[0])  # Assuming height is 0 for the polygon corners
            enu_xy = [enu[0], enu[1]]
            enu_coords.append(enu_xy)
        # enu_coords = np.array(enu_coords)
        print("ENU coordinates of polygon corners:", enu_coords)
        # print("ENU coords shape:", enu_coords.shape)

        # Create a Path object from the ENU coordinates of the polygon corners
        polygon_path = Path(enu_coords)

        # Find nodes within the polygon
        nodes_in_polygon = []
        for node_id, enu in self.node_enu_dict.items():
            if polygon_path.contains_point(enu):
                nodes_in_polygon.append(node_id)

        return nodes_in_polygon


if __name__ == '__main__':
    parser = ArgumentParser(description="Region Monitoring Test")
    parser.add_argument(
        '--params',
        type=str,
        help="/path/to/region_querying.yaml file of region monitoring tasks"
    )
    args = parser.parse_args()

    with open(args.params, 'rb') as f:
        region_task_params = yaml.safe_load(f)
    
    terra = load_terra(region_task_params["terra"])
    terra.visualizer.level_offset = 0.0  # Set level offset for better visualization
    
    #Load in the GPS and ENU dictionaries and the transformation from point cloud coordinates to ENU coordinates from file paths specified in the region_task_params
    node_gps_dict_path = region_task_params["gps_dict"]
    node_enu_dict_path = region_task_params["enu_dict"]
    transformation_path = region_task_params["transformation"]

    # Load the dictionaries and transformation pkl files from the file paths
    with open(node_gps_dict_path, 'rb') as f:
        node_gps_dict = pkl.load(f)
    with open(node_enu_dict_path, 'rb') as f:
        node_enu_dict = pkl.load(f)
    with open(transformation_path, 'rb') as f:
        transformation = pkl.load(f)

    R = transformation['rotation']
    scale = transformation['scale']
    t = transformation['translation']
    enu_origin_node_id = transformation['enu_origin_node_id']
    region_gps_visualizer = RegionGPSVisualizer(terra, node_gps_dict, node_enu_dict, R, scale, t, node_origin_id=enu_origin_node_id)


    ##### Marina P1 #####
    # # Red region
    # red_gps_coords = [
    #      [40.242157415183954, -111.7346451551257],
    #      [40.242050951019465, -111.73343547881638],
    #      [40.24089826098093, -111.73382439913756],
    #      [40.239643289794564, -111.73445337437707],
    #      [40.23961872024371, -111.73519366409187],
    #      [40.24048888640246, -111.73524194385587]
    # ]
    # red_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(red_gps_coords)
    # print(f"Red Nodes: {red_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     red_nodes
    # )

    # # Green region
    # green_gps_coords = [
    #     [40.24118547973413, -111.7458241491188],
    #     [40.240602527223636, -111.74022067757495],
    #     [40.24023808313468, -111.74049962732255],
    #     [40.24087074045152, -111.74589891408323]
    # ]
    # green_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(green_gps_coords)
    # print(f"Green Nodes: {green_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     green_nodes
    # )
    
    # # Blue region
    # blue_gps_coords = [
    #      [40.240074371345, -111.7385053467482],
    #      [40.23966283281242, -111.73881380079602],
    #      [40.23898511962814, -111.7382290792097],
    #      [40.23862652491201, -111.7373183546377],
    #      [40.239083114299675, -111.73682482816118]
    # ]
    # blue_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(blue_gps_coords)
    # print(f"Blue Nodes: {blue_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     blue_nodes
    # )

    # # Purple region
    # purple_gps_coords = [
    #     [40.23874220814309, -111.73691334109387],
    #     [40.238034796834725, -111.73732640128384],
    #     [40.23747889535013, -111.73817397934025],
    #     [40.23717381304281, -111.73798086028422],
    #     [40.237499370623546, -111.73740552642978],
    #     [40.23795801513077, -111.73689322451783],
    #     [40.23796313391349, -111.73685299138116],
    #     [40.23850674644134, -111.73653917292509],
    #     [40.23860604995924, -111.73661159257371]
    # ]
    # purple_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(purple_gps_coords)
    # print(f"Purple Nodes: {purple_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     purple_nodes
    # )

    # ##### Marina P2 #####
    # # Red region
    # red_gps_coords = [
    #     [40.241142735818855, -111.74638351027345],
    #     [40.240916153564505, -111.7473035102952],
    #     [40.236775267610255, -111.74346379114549],
    #     [40.236902892887194, -111.74084523156681],
    #     [40.237335348282556, -111.74092303448475],
    #     [40.23730122281264, -111.74317815418736],
    #     [40.23967455468688, -111.74506439598704]
    # ]
    # red_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(red_gps_coords)
    # print(f"Red Nodes: {red_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     red_nodes
    # )


    # ##### Rock Canyon Campground #####
    # # Red region
    # red_gps_coords = [
    #     [40.29119988211638, -111.60698558288246],
    #     [40.29095897669601, -111.6069661368664],
    #     [40.29099068828662, -111.60673412577825],
    #     [40.29104439336583, -111.60673814909191],
    #     [40.29109400659155, -111.60684543745637],
    #     [40.291187095226064, -111.60669523374614],
    #     [40.291231082118635, -111.6067267497032],
    #     [40.2912776263575, -111.6069205393115]
    # ]
    # red_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(red_gps_coords)
    # print(f"Red Nodes: {red_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     red_nodes
    # )

    # # Green region
    # green_gps_coords = [
    #     [40.29165060612899, -111.60819801239231],
    #     [40.29123247095061, -111.60784840111937],
    #     [40.29105601173413, -111.60751714829912],
    #     [40.29100997881833, -111.60732805255952],
    #     [40.2911951332394, -111.60718455437204],
    #     [40.29124525783098, -111.60727574947909],
    #     [40.29131379547587, -111.607442046444],
    #     [40.2914089297036, -111.60771697287794],
    #     [40.29165750560202, -111.60782292013783]
    # ]
    # green_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(green_gps_coords)
    # print(f"Green Nodes: {green_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     green_nodes
    # )
         


    ##### Nunns Park #####
    # # Red region
    # red_gps_coords = [
    #     [40.337579358319054, -111.60995247767303],
    #     [40.33762615217472, -111.61011547771135],
    #     [40.33747447542124, -111.61070185447248],
    #     [40.337145341809624, -111.61110184597922],
    #     [40.3373260491143, -111.61119286453216],
    #     [40.337711663976506, -111.61069120413572],
    #     [40.33781815137914, -111.61002867370148],
    #     [40.33776813459198, -111.60985298672719]
    # ]
    # red_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(red_gps_coords)
    # print(f"Red Nodes: {red_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     red_nodes
    # )

    # # Green region
    # green_gps_coords = [
    #     [40.33801176440014, -111.60931322548299],
    #     [40.33764712608347, -111.60949102916781],
    #     [40.33751643664956, -111.60929205837766],
    #     [40.33788107567252, -111.60877134758641],
    #     [40.33801297479992, -111.60909732417544]
    # ]
    # green_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(green_gps_coords)
    # print(f"Green Nodes: {green_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     green_nodes
    # )

    # # Blue region
    # blue_gps_coords = [
    #     [40.33684760957797, -111.61192253751024],
    #     [40.337035532198335, -111.61223929935859],
    #     [40.336630429666236, -111.61260882243998],
    #     [40.336462125631776, -111.61230149638554],
    #     [40.33651087786732, -111.61214552912546],
    #     [40.33641024815928, -111.61192738501165],
    #     [40.336534964555284, -111.61184960095503],
    #     [40.33667092560965, -111.6119783469798],
    #     [40.336728172287415, -111.61203601447008],
    #     [40.3368334651572, -111.61191665617626]
    # ]
    # blue_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(blue_gps_coords)
    # print(f"Blue Nodes: {blue_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     blue_nodes
    # )  

    # #Purple region
    # purple_gps_coords = [
    #     [40.336088233436314, -111.61262341824023],
    #     [40.335323808148196, -111.61285100011598],
    #     [40.33493788471448, -111.61343119392126],
    #     [40.3348706965385, -111.6136315160627],
    #     [40.33488138557099, -111.61369161270513],
    #     [40.33507226086578, -111.61359746129865],
    #     [40.3355775525413, -111.61332703759274],
    #     [40.33559281938573, -111.61343719278352],
    #     [40.3362920369194, -111.61312074694537]
    # ]
    # purple_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(purple_gps_coords)
    # print(f"Purple Nodes: {purple_nodes}")
    # terra.visualizer.display_selected_nodes(
    #     terra.terra_3dsg,
    #     purple_nodes
    # )


    ##### Provo River #####
    # Red region
    red_gps_coords = [
         [40.24181861674022, -111.68626048647668],
         [40.24161899529548, -111.6864361711735],
         [40.241525838419854, -111.68625512205847],
         [40.24173979193236, -111.6860418864341]
    ]
    red_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(red_gps_coords)
    print(f"Red Nodes: {red_nodes}")
    terra.visualizer.display_selected_nodes(
        terra.terra_3dsg,
        red_nodes,
        pc=terra.pc
    )

    # Green region
    green_gps_coords = [
        [40.24175412372867, -111.68605127416532],
        [40.242069422346525, -111.68669902765231],
        [40.24261658480747, -111.68622829996741],
        [40.24229412290208, -111.68558322868729],
    ]
    green_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(green_gps_coords)
    print(f"Green Nodes: {green_nodes}")
    terra.visualizer.display_selected_nodes(
        terra.terra_3dsg,
        green_nodes,
        pc=terra.pc
    )

    # Blue region
    blue_gps_coords = [
        [40.24145291029915, -111.68569179339279],
        [40.24135361096384, -111.68563814921058],
        [40.24108437595075, -111.68584870262585],
        [40.240936961649666, -111.68610485359599],
        [40.240843803835546, -111.6863261358477],
        [40.240990194628736, -111.68642135427115],
        [40.24112634801944, -111.68622421190145],
        [40.2412440744147, -111.68601768179985],
        [40.24138739322829, -111.68596001430397]
    ]
    blue_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(blue_gps_coords)
    print(f"Blue Nodes: {blue_nodes}")
    terra.visualizer.display_selected_nodes(
        terra.terra_3dsg,
        blue_nodes,
        pc=terra.pc
    )

    # Purple region
    purple_gps_coords = [
        [40.24219713883745, -111.68548928661745],
        [40.2419545232841, -111.68578835293339],
        [40.241867509055815, -111.6858674781055],
        [40.241828608540274, -111.6857575075286],
        [40.24178356581082, -111.68552683754501],
        [40.241868532752704, -111.68544637127167],
        [40.2417436415483, -111.68533237738443],
        [40.24181939525716, -111.68519022030152],
        [40.2419453100181, -111.68531628412975],
        [40.24204767956985, -111.68521972460174],
        [40.24214595419391, -111.68530019087508]
    ]
    purple_nodes = region_gps_visualizer.get_nodes_in_polygon_from_refs(purple_gps_coords)
    print(f"Purple Nodes: {purple_nodes}")
    terra.visualizer.display_selected_nodes(
        terra.terra_3dsg,
        purple_nodes,
        pc=terra.pc
    )
         
    
    