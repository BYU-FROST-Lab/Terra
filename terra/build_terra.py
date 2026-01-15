from argparse import ArgumentParser
import os
import yaml
from tqdm import tqdm
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pickle as pkl
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
import networkx as nx
import clip
import open3d as o3d
from gvd import DistanceMap
# from Terra import Terra
from scipy.cluster.hierarchy import fcluster
from utils import tensor_cosine_similarity, numeric_key, find_latest_itr

from visualize_terra import Terra_Visualizer

class Terra_Builder:
    def __init__(self, args):
        ## Load arguments and parameters
        self.data_folder = args['data_folder']
        self.output_folder = args['output_folder']
        self.save_average_embeddings = args['save_average_embeddings']
        self.DEBUG_MODE = args['debug']
        self.cam2pt_dist_thresh = args['cam2point_dist_threshold']
        if self.DEBUG_MODE:
            print("Running in DEBUG_MODE")
        self.terrain_classes = args['terrain_classes']
        self.terrain_threshold = args['terrain_threshold']
        self.cossim_weight_ratio = args['cossim_weight_ratio']
        self.gvd_params = {
            'og_res': args['og_resolution'],
            'max_itr': args['max_iterations'],
            'max_dev': args['max_deviation'],
            'max_dist': args['max_distance']
        }
        self.z_offset_viz = args['z_offset_viz']
        self.z_offset_lidar = args['z_offset_lidar']
        self.region_method = args['region_method']
        if self.region_method == "agglomerative":
            self.hierarchical_distances = args['agg_hierarchical_distances']
        elif self.region_method == "spectral":
            self.min_graph_area = args['spec_min_graph_area']
            self.max_sem_diff = args['spec_max_semantic_diff']            
            
        ## Initialize Clip Model ##
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, clip_preprocess = clip.load("ViT-B/16", device=self.device)

        ## Load in Data ##
        global_pc_folder = os.path.join(self.data_folder, "global_pc")
        global_pc_files = sorted(Path(global_pc_folder).glob("*.npy"),key=numeric_key)        
        latest_global_pc_file = global_pc_files[-1] # global_map_idx 33 for sunny midday data
        self.global_pc = np.load(latest_global_pc_file) # (num_pts,4)

        self.last_itr = find_latest_itr(self.output_folder)
        self.clip_tensor = torch.load(self.output_folder+f"/ptxpt_clip_tensor_itr{self.last_itr}.pt") # (num_clip_ids,512)
        with open(self.output_folder+f"/ptxpt_pc_dict_itr{self.last_itr}.pkl", "rb") as f:
            self.pc_dict = pkl.load(f) # {global_pt_idx: {clip_id: count, ...}, ...}
        self.img_clip_tensor = torch.load(self.output_folder+f"/img_clip_tensor_itr{self.last_itr}.pt") # (num_imgs,512)
        with open(self.output_folder+f"/ptxpt_gidx2imgidx_{self.cam2pt_dist_thresh}m_dist_dict_itr{self.last_itr}.pkl", "rb") as f:
            self.map_globalidx2imgidx = pkl.load(f) # {global_pt_idx: set{img_idx0, ...}, ...}
        with open(self.output_folder+f"/ptxpt_gidx2imgidx_no_dist_dict_itr{self.last_itr}.pkl", "rb") as f:
            self.map_globalidx2imgidx_nodist = pkl.load(f) # {global_pt_idx: set{img_idx0, ...}, ...}
        with open(self.output_folder+f"/saved_img_names_itr{self.last_itr}.pkl", "rb") as f:
            self.img_names = pkl.load(f)
        print("Finished loading CLIP and semantics for point cloud")

    def build_3dsg(self):
        ## Terrain Clip Embeddings ##
        self.gen_terrain_clip_embeddings()

        ## Calculate/load average Clip embeddings for each global point ##
        if self.save_average_embeddings:
            self.avg_clip_embeddings()     
        else:
            #Load in data
            self.load_or_save_avg_embeddings(save=False)
        
        ## Terrain GVDs ##
        self.build_terrain_gvds()

        ## Add Edges between GVDs
        self.add_edges_between_terrain_gvds()

        ## Associate Image CLIP Embs to GVD Nodes
        self.associate_image_clips_to_gvd_nodes()

        ## Convert GVD to nx.Graph ##
        self.convert_GVD_to_graph()

        ## Build Hierarchical Regions
        self.build_hierarchical_regions()

        ## Save 3DSG ##
        self.save_3dsg()

        # self.terra = Terra()
        # self.terra.set_Terra_3DSG(self.TERRA_3DSG)

        print("Done building 3DSG!")

    def gen_terrain_clip_embeddings(self):   
        # Embed user input with CLIP
        self.input_terrain_clip_embs = [self.clip_model.encode_text(clip.tokenize([terrain]).to(self.device)).float() for terrain in self.terrain_classes]
        self.input_terrain_clip_tensor = torch.vstack(self.input_terrain_clip_embs) # (num_input_terrain, 512)

    def avg_clip_embeddings(self):
        print("Begin averaging CLIP embeddings for each global point...")
        self.global_clip_filt = torch.zeros((0,512),device=self.device) # (num_global_pts, 512)
        self.terrain_ids = {} # Dictionary of terrain index to global point indices classified as that terrain
        self.nonsemantic_gidx = []
        self.terrain_gidx = []
        self.nonterrain_gidx = []
        self.semantic_gidxs = []
        t0_clipavg = time.time()
        max_prompt_scores = []
        for global_idx in tqdm(range(self.global_pc.shape[0]), desc="Processing points"):
            if global_idx in self.pc_dict.keys():
                num_detections = 0
                clip_emb_vec = torch.zeros((1,512),device=self.device)
                for clip_id, count in self.pc_dict[global_idx].items():
                    clip_emb_vec += count * self.clip_tensor[clip_id,:]
                    num_detections += count
                clip_avg_emb = clip_emb_vec / num_detections
                self.global_clip_filt = torch.concatenate((self.global_clip_filt, clip_avg_emb))
                
                self.semantic_gidxs.append(global_idx)

                # Score how close the avg clip embedding is to each terrain class
                scores = tensor_cosine_similarity(
                                clip_avg_emb, 
                                self.input_terrain_clip_tensor) # (num_clusters, num_prompt_tasks)
                max_prompt_score = scores.max().item()
                max_prompt_scores.append(max_prompt_score)
                max_prompt_score_idx = scores.argmax().item()

                if max_prompt_score > self.terrain_threshold:
                    self.terrain_gidx.append(global_idx)
                    if max_prompt_score_idx not in self.terrain_ids.keys():
                        self.terrain_ids[max_prompt_score_idx] = [global_idx]
                    else:
                        self.terrain_ids[max_prompt_score_idx].append(global_idx)
                else:
                    self.nonterrain_gidx.append(global_idx)
            else:
                self.nonsemantic_gidx.append(global_idx)    
        t1_clipavg = time.time()    
        print(f"Finished averaging CLIP embedings for each global point in {t1_clipavg-t0_clipavg} seconds")

        ## Save the Data ##
        self.load_or_save_avg_embeddings(save=True)
        
        if self.DEBUG_MODE:
            self.plot_score_dist(max_prompt_scores)

    def load_or_save_avg_embeddings(self, save=False):
        # Get Paths to save data   
        global_clip_path = os.path.join(self.output_folder, "global_clip_filt.pt")
        semantic_gidxs_path = os.path.join(self.output_folder, "semantic_gidxs.pkl")
        terrain_gidx_path = os.path.join(self.output_folder, "terrain_gidx.pkl")
        terrain_ids_path = os.path.join(self.output_folder, "terrain_ids.pkl")
        nonterrain_gidx_path = os.path.join(self.output_folder, "nonterrain_gidx.pkl")

        if save:
            ## Save average embeddings and global to semantic index map and terrain points
            torch.save(self.global_clip_filt, global_clip_path)
            with open(semantic_gidxs_path, "wb") as f: pkl.dump(self.semantic_gidxs, f)
            with open(terrain_gidx_path, "wb") as f: pkl.dump(self.terrain_gidx, f)
            with open(nonterrain_gidx_path, "wb") as f: pkl.dump(self.nonterrain_gidx, f)
            with open(terrain_ids_path, "wb") as f: pkl.dump(self.terrain_ids, f)
        
        else:
            print("Loading saved CLIP semantic results...")
            self.global_clip_filt = torch.load(global_clip_path)
            with open(terrain_ids_path, "rb") as f: self.terrain_ids = pkl.load(f)   

    def plot_score_dist(self, scores):
        # Plot distribution of max prompt scores
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=50)
        plt.axvline(
            self.terrain_threshold,
            linestyle='--',
            linewidth=2,
            label=f"Terrain threshold = {self.terrain_threshold}"
        )
        plt.xlabel("Max CLIP cosine similarity to terrain prompts")
        plt.ylabel("Number of global points")
        plt.title("Distribution of Max Terrain Prompt Scores")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def build_terrain_gvds(self):
        print("\nGenerating Terrain GVDs")

        # For each terrain type, 
        # grab the x,y coord of all global points classified as that terrain    
        t0_terrain_gvd = time.time() 
        terrain_list = []
        for t_idx in range(len(self.terrain_classes)):
            if t_idx in self.terrain_ids:
                terrain_list.append(self.global_pc[self.terrain_ids[t_idx],:2])
            else:
                terrain_list.append(np.array([[0,0]]))
        terrain_tuple = tuple(terrain_list)       
        terrain_pts = np.vstack(terrain_tuple) #(N, 2)
        
        self.x_min, self.y_min = np.min(terrain_pts, axis=0)
        x_max, y_max = np.max(terrain_pts, axis=0)
        
        grid_width = int(np.ceil((x_max - self.x_min) / self.gvd_params['og_res']))
        grid_height = int(np.ceil((y_max - self.y_min) / self.gvd_params['og_res']))
        self.dist_maps = []
        for t_idx, xy_pts in enumerate(terrain_tuple):
            print(f"Building GVD for {self.terrain_classes[t_idx]} terrain")

            # Create a binary occupancy grid (0-> terrain point, 1-> non-terrain point)
            grid = np.ones((grid_height, grid_width), dtype=int)
            x_indices = ((xy_pts[:, 0] - self.x_min) / self.gvd_params['og_res']).astype(int)
            y_indices = ((xy_pts[:, 1] - self.y_min) / self.gvd_params['og_res']).astype(int)
            grid[y_indices, x_indices] = 0  # Use y first to match (row, col) indexing
            
            #############################
            ## Run brushfire algorithm ##
            #############################
            structure = np.ones((3,3), dtype=bool)
            eroded_arr = binary_erosion(grid[::-1,:].astype(bool), structure=structure)
            final_arr = binary_dilation(eroded_arr, structure=structure)
            
            #Build distance map and GVD
            dm = DistanceMap(final_arr.astype(int))
            gvd_t0 = time.time()
            dm.compute_static()
            gvd_t1 = time.time()
            # print("Finished building GVD for terrain in",gvd_t1-gvd_t0,"seconds")
            gvd_t2 = time.time()
            dm.prune_4connected()
            gvd_t3 = time.time()
            # print("Finished pruning GVD for terrain in",gvd_t3-gvd_t2,"seconds")
            
            gvd_t4 = time.time()
            dm.floodfill_split_technique(
                self.gvd_params['max_itr'], 
                self.gvd_params['max_dev'], 
                self.gvd_params['max_dist'])
            gvd_t5 = time.time()
            # print("Finished flood-fill and split technique in",gvd_t5-gvd_t4,"seconds")        
            if self.DEBUG_MODE:
                dm.display()
                plt.title(f"DM for {self.terrain_classes[t_idx]} Terrain")
                dm.display(plot_gvd=True)
                plt.title(f"GVD for {self.terrain_classes[t_idx]} Terrain")
                dm.display(plot_flood_fill=True)
                plt.title(f"Flood-Fill for {self.terrain_classes[t_idx]} Terrain")
                dm.display(plot_node_edges=True)
                plt.title(f"GVD Nodes and Edges for {self.terrain_classes[t_idx]} Terrain")
                plt.show()
            self.dist_maps.append(dm)
        t1_terrain_gvd = time.time()
        print("Runtime to make GVDs",t1_terrain_gvd - t0_terrain_gvd,"seconds")

    def add_edges_between_terrain_gvds(self):
        #################################################
        ## ------- Add edges between GVDs -------      ##
        ## 1) Grab dead-end nodes (== 1 connection)    ##
        ##    from each terrain GVD                    ##
        ## 2) Connect dead-end nodes to closest node   ##
        #     in other terrain if within a threshold   ##
        #################################################
        print("\nAdding edges between terrain")
        t0_combine_gvds = time.time()
        terrain_deadend_nodes = {}
        for terrain_idx, dm in enumerate(tqdm(self.dist_maps, desc="Adding edges between same terrain")):
            terrain_deadend_nodes[terrain_idx] = []
            connected_node_id_counts = dm.get_connected_node_id_counts()
            for node_id, count in connected_node_id_counts.items():
                if count <= 2:
                    y, x = dm.gvd_node_ids2coords[node_id]
                    terrain_deadend_nodes[terrain_idx].append(
                        (y, x)     
                    )
                    
        ## Display Dead-End Nodes ontop of Normal GVD Nodes
        if self.DEBUG_MODE:
            for terrain_idx, dm in enumerate(self.dist_maps):
                fig, ax = dm.display(plot_node_edges=True)
                for node_coords in terrain_deadend_nodes[terrain_idx]:
                    circle = plt.Circle((node_coords[1],node_coords[0]),1,color=[1,1,1],alpha=1.0)
                    ax.add_patch(circle)
                plt.title(f"Displaying Dead-End Nodes for {self.terrain_classes[terrain_idx]} Terrain")
                plt.show()

        ## Connect Dead-End Nodes to closest node in diff. terrain (takes a long time in debug mode)
        self.new_terrain_connections = []
        connecting_terrains = []
        for terrain_idx in tqdm(terrain_deadend_nodes.keys(), desc="Connecting dead-end nodes"):
            for dy,dx in terrain_deadend_nodes[terrain_idx]:
                min_dist = np.inf
                min_coord = None
                min_terrain_idx = -1
                for dm_idx, dm in enumerate(self.dist_maps):        
                    if dm_idx == terrain_idx:
                        continue
                    for y in range(dm.gvd_nodes.shape[0]):
                        for x in range(dm.gvd_nodes.shape[1]):
                            if dm.gvd_nodes[y,x]:
                                dist = np.linalg.norm(np.array([y,x])-np.array([dy,dx]))
                                if dist < min_dist:
                                    min_dist = dist
                                    min_coord = (y,x)
                                    min_terrain_idx = dm_idx
                if min_dist < (2*self.gvd_params['max_dist']):
                    connecting_terrains.append((terrain_idx, min_terrain_idx))
                    self.new_terrain_connections.append([(dy,dx), min_coord])

        # Convert GVD Nodes to real-world coordinates
        distinct_colors = [[216/255,27/255,96/255],[0,1,0],[0,0,1],[0.5,0.5,0.5],[1,164/255,0],[0,1,1],[255./255.,51./255.,255./255.]]
        self.terrain_nodes = []
        self.terrain_ids_for_nodes = []
        self.terrain_coord_to_idx = {}
        node_idx = 0
        for terrain_idx, dm in enumerate(tqdm(self.dist_maps, desc="Converting node positions to real world coords")):
            for y in range(dm.gvd_nodes.shape[0]):
                for x in range(dm.gvd_nodes.shape[1]):
                    if dm.gvd_nodes[y,x]:
                        # Convert gvd_node coords from voxels back to real world units [m]
                        y_real = (dm.gvd_nodes.shape[0] - y) * self.gvd_params['og_res'] + self.y_min
                        x_real = x * self.gvd_params['og_res'] + self.x_min

                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate([x_real,y_real,self.z_offset_viz])
                        sphere.paint_uniform_color(distinct_colors[terrain_idx])
                        self.terrain_nodes.append(sphere)
                        self.terrain_ids_for_nodes.append(terrain_idx)
                        
                        self.terrain_coord_to_idx[(y,x)] = node_idx
                        node_idx += 1
        t1_combine_gvds = time.time()
        print("Runtime to combine GVD edges",t1_combine_gvds - t0_combine_gvds,"seconds")

    def associate_image_clips_to_gvd_nodes(self):
        print("\nAssociating image CLIP embeddings to each GVD Node")
        t0_associate_clip_emb = time.time()
        global_kdtree = KDTree(self.global_pc[:, :3])  # Using the x, y columns
        self.map_gvdnodes2clipembs = {}
        self.map_nodeid2imgidx = {}
        local_img_cntr = 0
        other_img_cntr = 0
        already_displayed=False
        for node_idx, sphere in enumerate(self.terrain_nodes):
            chosen_g_idx = -1
            
            # Find closest global_idx node
            xyz = sphere.get_center()
            xyz[2] = self.z_offset_lidar # shifts node point to be on ground
            dist, g_idxs = global_kdtree.query(xyz, k=20) # 20 closest neighboring pts
            
            found = False
            for g_idx in g_idxs:        
                if g_idx in self.map_globalidx2imgidx:        
                    img_indices = list(self.map_globalidx2imgidx[g_idx])
                    local_img_cntr += 1
                    chosen_g_idx = g_idx
                    found = True
                    break
                elif g_idx in self.map_globalidx2imgidx_nodist:
                    img_indices = list(self.map_globalidx2imgidx_nodist[g_idx])
                    other_img_cntr += 1
                    chosen_g_idx = g_idx
                    found = True
                    break
            if not found:
                assert False, f"Image association not found"
            
            if self.DEBUG_MODE and not already_displayed:
                print("Matching Global Index:", chosen_g_idx)
                num_disp_imgs = min(len(img_indices), 20) # disp max 20 images
                for cntr, img_idx in enumerate(img_indices):
                    if cntr < num_disp_imgs:
                        plt.figure()
                        plt.imshow(plt.imread(self.img_names[img_idx]))
                    else:
                        break
                plt.show()
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.global_pc[:,:3])
                pcd.paint_uniform_color([0.5,0.5,0.5])            
                global_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate(self.global_pc[chosen_g_idx,:3])
                global_sphere.paint_uniform_color([1,0,0])
                places_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate(xyz)
                places_sphere.paint_uniform_color([0,0,1])
                o3d.visualization.draw_geometries([pcd]+[global_sphere]+[places_sphere])
                already_displayed = True
                
            self.map_nodeid2imgidx[node_idx] = img_indices
            
            clip_emb_vec = torch.mean(self.img_clip_tensor[img_indices,:], dim=0, keepdim=True)
                    
            self.map_gvdnodes2clipembs[node_idx] = clip_emb_vec    
        # print(f"20m image matches {local_img_cntr}, other image matches {other_img_cntr}")
        t1_associate_clip_emb = time.time()
        print("Runtime to associate CLIP Embeddings to GVD nodes",t1_associate_clip_emb - t0_associate_clip_emb,"seconds")

    def convert_GVD_to_graph(self):
        print("\nConverting GVDs into a nx.Graph")
        t0_placesgraph = time.time()
        terra_graph = nx.Graph()
        for node_idx, sphere in enumerate(self.terrain_nodes):
            xy = sphere.get_center()[:2]
            terra_graph.add_node(
                node_idx,
                level=1, # level 0=objects; 1=terrain; 2+=regions
                pos=xy,
                embedding=self.map_gvdnodes2clipembs[node_idx],
                terrain_id=self.terrain_ids_for_nodes[node_idx],
            )
        
        # Connect similar terrain nodes following GVD
        for terrain_idx, dm in enumerate(self.dist_maps):
            connected_node_coords = dm.get_connected_node_coords() 
            for (n1_coord, n2_coord) in connected_node_coords:
                n1_id = self.terrain_coord_to_idx[n1_coord]
                n2_id = self.terrain_coord_to_idx[n2_coord]
                
                # Add in edge weights
                wij = self.edge_weight(terra_graph, n1_id, n2_id)
                terra_graph.add_edge(n1_id, n2_id, weight=wij)
        
        # Add in dead-end node edges between terrain types
        for (n1_coord, n2_coord) in self.new_terrain_connections:
            n1_id = self.terrain_coord_to_idx[n1_coord]
            n2_id = self.terrain_coord_to_idx[n2_coord]

            # Add in edge weights
            wij = self.edge_weight(terra_graph, n1_id, n2_id)
            terra_graph.add_edge(n1_id, n2_id, weight=wij)
        t1_placesgraph = time.time()
        print("Runtime to build nx.Graph from GVDs",t1_placesgraph - t0_placesgraph,"seconds")
        
        self.terra_graph = self.get_largest_components(terra_graph)[0] # removes disconnected components (needed for hier-regions)
        
    def build_hierarchical_regions(self):
        print("Building hierarchical regions...")
        t0_cluster = time.time()
        if self.region_method == "agglomerative":
            self.build_agglomerative_regions()
            t1_cluster = time.time()
            print(f"Time for agglomerative clustering: {t1_cluster - t0_cluster} seconds")
        elif self.region_method == "spectral":
            self.build_spectral_regions()
            t1_cluster = time.time()
            print(f"Time for spectral clustering: {t1_cluster - t0_cluster} seconds")
        else:
            assert False, "Unidentified region method specified"
        
    def build_agglomerative_regions(self):
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=50, metric='precomputed', linkage='average')
        
        terra_graph_dense = self.terra_graph.copy()
        t0_fc = time.time()
        for n1_id in list(self.terra_graph.nodes):
            for n2_id in list(self.terra_graph.nodes):
                if n1_id == n2_id:
                    continue
                if not terra_graph_dense.has_edge(n1_id, n2_id):
                    wij = self.edge_weight(terra_graph_dense, n1_id, n2_id)
                    terra_graph_dense.add_edge(n1_id, n2_id, weight=wij)
        print(f"Time to make fully connected graph: {time.time() - t0_fc} seconds")
        
        nodes = list(self.terra_graph.nodes)
        map_idx2nodeidx = {}
        prev_id = len(self.terrain_nodes)
        for i, n_id in enumerate(nodes):
            map_idx2nodeidx[i] = n_id
        
        dist_matrix = nx.to_numpy_array(terra_graph_dense, weight='weight')
        agg_model = ac.fit(dist_matrix)
        
        counts = np.zeros(agg_model.children_.shape[0])
        n_samples = len(agg_model.labels_)
        for i, merge in enumerate(agg_model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [agg_model.children_, agg_model.distances_, counts]
        ).astype(float)
        
        # Now extract the three hierarchical levels:
        hierarchical_clusters = []
        for dist in self.hierarchical_distances:
            hierarchical_clusters.append(
                fcluster(linkage_matrix, t=dist, criterion='distance')
            )
        
        for layer_idx, labels in enumerate(hierarchical_clusters):
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if counts.size == 0 or counts.max() == 0:
                print("No clusters found, skipping this mask.")
                continue
            for l in unique_labels:
                curr_label_indices = np.where(labels == l)[0]
                
                prev_id += 1
                pos = np.array([self.terra_graph.nodes[map_idx2nodeidx[idx]]["pos"] for idx in curr_label_indices])
                avg_pos = np.mean(pos,axis=0) # [2,]
                embs = torch.vstack([self.terra_graph.nodes[map_idx2nodeidx[idx]]['embedding'] for idx in curr_label_indices])
                avg_emb = torch.mean(embs,dim=0).unsqueeze(0) # [1,512]
                
                # Add region node
                self.terra_graph.add_node(
                    prev_id,
                    pos=avg_pos,
                    embedding=avg_emb,
                    terrain_id=-1,
                    level=layer_idx+2,
                )
                
                # Recursively add edges to child nodes of region node
                if layer_idx == 0:
                    for idx in curr_label_indices:
                        child_id = map_idx2nodeidx[idx]
                        wij = self.edge_weight(self.terra_graph, prev_id, child_id)
                        self.terra_graph.add_edge(prev_id, child_id, weight=wij)
                else:
                    added_edge_nodes = []
                    for idx in curr_label_indices:
                        a_child_node = map_idx2nodeidx[idx]
                        childs_layer = 1
                        while True:
                            stop = True
                            connected_nodes = list(self.terra_graph.neighbors(a_child_node))
                            for cn_id in connected_nodes:
                                if self.terra_graph.nodes[cn_id]["level"] > childs_layer and \
                                    self.terra_graph.nodes[cn_id]["level"] == (layer_idx+1) and \
                                    cn_id not in added_edge_nodes:
                                    
                                    a_child_node = cn_id
                                    wij = self.edge_weight(self.terra_graph, prev_id, a_child_node)
                                    self.terra_graph.add_edge(prev_id, a_child_node, weight=wij)
                                    added_edge_nodes.append(cn_id)
                                    stop = True
                                    break
                                elif self.terra_graph.nodes[cn_id]["level"] > childs_layer and cn_id not in added_edge_nodes:
                                    a_child_node = cn_id
                                    childs_layer = self.terra_graph.nodes[cn_id]["level"]
                                    stop = False
                            if stop:
                                break
        # Finally, add a global root node if multiple top-level regions exist
        self.add_root_region_node(prev_id+1)
    
    def add_root_region_node(self, root_node_id):
        root_node_level = len(self.hierarchical_distances) + 2

        # Find top-level region nodes (no parents at same or higher level)
        top_level_nodes = [n_id for n_id in list(self.terra_graph.nodes) \
            if self.terra_graph.nodes[n_id]["level"] == root_node_level - 1
        ]
        
        if len(top_level_nodes) <= 1:
            print("Only one top-level region — no root needed.")
            return

        # Create root node
        positions = np.array([self.terra_graph.nodes[n]["pos"] for n in top_level_nodes])
        avg_pos = positions.mean(axis=0)

        embs = torch.vstack([self.terra_graph.nodes[n]["embedding"] for n in top_level_nodes])
        avg_emb = torch.mean(embs, dim=0).unsqueeze(0)

        self.terra_graph.add_node(
            root_node_id,
            level=root_node_level,
            pos=avg_pos,
            embedding=avg_emb,
            terrain_id=-1,
        )

        # Connect root to all top-level regions
        for n in top_level_nodes:
            w = self.edge_weight(self.terra_graph, root_node_id, n)
            self.terra_graph.add_edge(root_node_id, n, weight=w)

        print(f"Added global root node {root_node_id} connecting {len(top_level_nodes)} regions.")

    
    def build_spectral_regions(self):
        terra_graph_orig = self.terra_graph.copy()
        
        subgraphs = []
        subgraph_hierarchy = {}
        graph_id_counter = len(self.terrain_nodes) + 1
        graphs_to_split = [(graph_id_counter, 0, None, terra_graph_orig, 0)]  # (id, level, parent_id, graph, cutoff)
        graph_id_counter += 1
        cutoff_dict = {
            0: "Too small",
            1: "Too semantically different",
            2: "Too deep of hierarchy"
        }
        while len(graphs_to_split) > 0:
            curr_id, level, parent_id, G_curr, cutoff = graphs_to_split.pop(0)

            # Track this split
            subgraph_hierarchy[curr_id] = {
                "level": level,
                "parent": parent_id,
                "child": [],
                "graph": G_curr,
                "stop_thresh": cutoff,
            }
            
            # Semantic affinity matrix (existing edge weights)
            semantic_aff = nx.to_numpy_array(G_curr, weight='weight')
            # Spectral clustering
            sc = SpectralClustering(2, affinity='precomputed', assign_labels='discretize')
            labels = sc.fit_predict(semantic_aff)

            nodes = list(G_curr.nodes)
            G1_curr = G_curr.subgraph(
                [nodes[i] for i in np.where(labels == 0)[0]]
            )
            G2_curr = G_curr.subgraph(
                [nodes[i] for i in np.where(labels == 1)[0]]
            )
            
            G1_size = self.get_graph_area(G1_curr)
            G2_size = self.get_graph_area(G2_curr) 

            if len(list(G1_curr.nodes)) < 4:
                # print(f"Finished at depth {level+1} b/c graph only has {len(list(G1_curr.nodes))} nodes")
                # print("  Saving G1")
                subgraph_hierarchy[curr_id]["child"].append(graph_id_counter)
                subgraph_hierarchy[graph_id_counter] = {
                    "level": level+1,
                    "parent": curr_id,
                    "child": [],
                    "graph": G1_curr,
                    "stop_thresh": cutoff,
                }
                graph_id_counter += 1
                subgraphs.append(G1_curr)
            elif G1_size < self.min_graph_area:
                ## Check semantic differences
                sem_diff = self.get_semantic_diff(G1_curr)
                # print("Semantic difference:",sem_diff)
                if sem_diff > self.max_sem_diff:
                    # print("Continuing b/c semantically not homogenous")
                    subgraph_hierarchy[curr_id]["child"].append(graph_id_counter)
                    graphs_to_split.append((graph_id_counter, level + 1, curr_id, G1_curr, 1))
                    graph_id_counter += 1
                else:           
                    # print(f"Finished at depth {level+1} b/c min graph area was reached and is homogenous")
                    # print("  Saving G1")
                    subgraph_hierarchy[curr_id]["child"].append(graph_id_counter)
                    subgraph_hierarchy[graph_id_counter] = {
                        "level": level+1,
                        "parent": curr_id,
                        "child": [],
                        "graph": G1_curr,
                        "stop_thresh": cutoff,
                    }
                    graph_id_counter += 1
                    subgraphs.append(G1_curr)
            else:
                subgraph_hierarchy[curr_id]["child"].append(graph_id_counter)
                graphs_to_split.append((graph_id_counter, level + 1, curr_id, G1_curr, cutoff))
                graph_id_counter += 1
            
            
            if len(list(G2_curr.nodes)) < 4:
                # print(f"Finished at depth {level+1} b/c graph only has {len(list(G2_curr.nodes))} nodes")
                # print("  Saving G2")
                subgraph_hierarchy[curr_id]["child"].append(graph_id_counter)
                subgraph_hierarchy[graph_id_counter] = {
                    "level": level+1,
                    "parent": curr_id,
                    "child": [],
                    "graph": G2_curr,
                    "stop_thresh": cutoff,
                }
                graph_id_counter += 1
                subgraphs.append(G2_curr)
            elif G2_size < self.min_graph_area:
                ## Check semantic differences
                sem_diff = self.get_semantic_diff(G2_curr)
                # print("Semantic difference:",sem_diff)
                if sem_diff > self.max_sem_diff:
                    # print("Continuing b/c semantically not homogenous")
                    subgraph_hierarchy[curr_id]["child"].append(graph_id_counter)
                    graphs_to_split.append((graph_id_counter, level + 1, curr_id, G2_curr, 1))
                    graph_id_counter += 1
                else:           
                    # print(f"Finished at depth {level+1} b/c min graph area was reached and is homogenous")
                    # print("  Saving G2")
                    subgraph_hierarchy[curr_id]["child"].append(graph_id_counter)
                    subgraph_hierarchy[graph_id_counter] = {
                        "level": level+1,
                        "parent": curr_id,
                        "child": [],
                        "graph": G2_curr,
                        "stop_thresh": cutoff,
                    }
                    graph_id_counter += 1
                    subgraphs.append(G2_curr)
            else:
                subgraph_hierarchy[curr_id]["child"].append(graph_id_counter)
                graphs_to_split.append((graph_id_counter, level + 1, curr_id, G2_curr, cutoff))
                graph_id_counter += 1
        '''
        subgraph_hierarchy[graph_id_counter] = {
                    "level": level+1,
                    "parent": curr_id,
                    "graph": G2_curr,
                    "stop_thresh": -1,
                }
        '''
        levels = defaultdict(list)
        for g_id, info in subgraph_hierarchy.items():
            levels[info["level"]].append((g_id, info["parent"], info["child"], info["graph"], info["stop_thresh"]))
        # Sort levels by depth
        sorted_levels = sorted(levels.items())
        max_lvl = max(levels.keys())
        for level, graphs in sorted_levels:
            for g_id, parent_id, children_ids, g, stop_thresh in graphs:
                pos = np.array([g.nodes[n_id]["pos"] for n_id in g.nodes])
                avg_pos = np.mean(pos,axis=0)
                
                embs = torch.vstack([g.nodes[n]['embedding'] for n in g.nodes])
                avg_emb = torch.mean(embs,dim=0).unsqueeze(0) # [1,512]
                
                # Add region node
                self.terra_graph.add_node(
                    g_id,
                    level=(max_lvl-level)+2,
                    pos=avg_pos,
                    embedding=avg_emb,
                    terrain_id=-1,
                )
                
                if len(children_ids) == 0: # leaf node
                    ## Connect to terrain nodes in nx.Graph...
                    for n_id in g.nodes:
                        wij = self.edge_weight(self.terra_graph, g_id, n_id)
                        self.terra_graph.add_edge(g_id,n_id,weight=wij)
                    if parent_id is not None:
                        wij = self.edge_weight(self.terra_graph, g_id, parent_id)
                        self.terra_graph.add_edge(g_id,parent_id,weight=wij)
                else: # region node
                    if parent_id is not None:
                        wij = self.edge_weight(self.terra_graph, g_id, parent_id)
                        self.terra_graph.add_edge(g_id,parent_id,weight=wij)
            
    def save_3dsg(self):
        with open(self.output_folder+f"/terra_3dsg_{self.cam2pt_dist_thresh}mimgembs_nodist1img_{self.region_method}cluster.pkl", "wb") as f:
            pkl.dump(self.terra_graph, f)
        with open(self.output_folder+f"/map_nodeid2imgidx_nodist1img_{self.region_method}cluster.pkl", "wb") as f:
            pkl.dump(self.map_nodeid2imgidx, f)
        print("Finished! 3DSG Saved!")

    def edge_weight(self, G, n1_id, n2_id):  
        cos_sim_places = tensor_cosine_similarity(G.nodes[n1_id]["embedding"], G.nodes[n2_id]["embedding"]).item()
        # cos_sim_terrain = tensor_cosine_similarity(
        #     terrain_embeddings[G.nodes[n1_id]["terrain_id"]],
        #     terrain_embeddings[G.nodes[n2_id]["terrain_id"]]
        # ).item()
        
        euclidean_dist = np.linalg.norm(G.nodes[n1_id]["pos"] - G.nodes[n2_id]["pos"]) 

        paper_weight = euclidean_dist + self.cossim_weight_ratio * (1 - cos_sim_places)
        return paper_weight

    @staticmethod
    def get_largest_components(G: nx.Graph, n=1):
        # Get all connected components sorted by size
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        
        # Extract top N components
        largest_components = components[:n]
        
        # Create subgraphs (use copy() to avoid view limitations)
        return [G.subgraph(comp).copy() for comp in largest_components]
    
    @staticmethod    
    def get_graph_area(G: nx.Graph):
        positions = np.array([G.nodes[n]['pos'] for n in G.nodes])

        # Compute min and max for each axis
        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)

        # Spatial size along each dimension
        spatial_size = max_coords - min_coords
        return spatial_size[0] * spatial_size[1]

    @staticmethod
    def get_semantic_diff(G: nx.Graph):
        place_node_embeddings = torch.vstack([G.nodes[n]['embedding'] for n in G.nodes])
        similarities = tensor_cosine_similarity(place_node_embeddings, place_node_embeddings)
        max_diff = 1 - torch.min(similarities)
        return max_diff.item()

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--params', 
                        type=str,  
                        help='YAML file of Build Terra arguments')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    with open(args.params, 'r') as file:
        build_terra_args = yaml.safe_load(file)
    terra_builder = Terra_Builder(build_terra_args)
    terra_builder.build_3dsg()
