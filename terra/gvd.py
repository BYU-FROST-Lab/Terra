import numpy as np
import heapq
from collections import deque
import matplotlib.pyplot as plt

class DistanceMap:
    def __init__(self, grid):
        self.grid = grid
        self.height, self.width = grid.shape
        self.dist_map = np.full_like(grid, np.inf, dtype=float)
        self.obst_map = np.full((self.height, self.width, 2), -1, dtype=int)
        self.voro_map = np.zeros_like(grid, dtype=bool)
        self.gvd_nodes = np.zeros_like(grid,dtype=bool)
        self.gvd_edges = np.zeros_like(grid,dtype=bool)
        self.gvd_ids = np.full_like(grid,-1,dtype=int)
        self.gvd_node_ids2coords = dict()
        
    def check_voro(self,y,x,ny,nx):
        """Implements the Voronoi condition from the paper."""
        if (self.dist_map[y, x] > 1 or self.dist_map[ny, nx] > 1) and \
           (self.obst_map[ny, nx][0] != -1) and \
           (not np.array_equal(self.obst_map[y, x], self.obst_map[ny, nx])) and \
           (not any(np.array_equal(self.obst_map[y, x], neighbor) for neighbor in self.get_8_neighbors(self.obst_map[ny, nx][0], self.obst_map[ny, nx][1]))):
            
            ## 4-connected approach
            if np.linalg.norm([y - self.obst_map[ny, nx][0], x - self.obst_map[ny, nx][1]]) <= np.linalg.norm([ny - self.obst_map[y, x][0], nx - self.obst_map[y, x][1]]):
                self.voro_map[y, x] = True
            if np.linalg.norm([ny - self.obst_map[y, x][0], nx - self.obst_map[y, x][1]]) <= np.linalg.norm([y - self.obst_map[ny, nx][0], x - self.obst_map[ny, nx][1]]):
                self.voro_map[ny, nx] = True
            
            # ## 8-connected approach
            # if np.linalg.norm([y - self.obst_map[ny, nx][0], x - self.obst_map[ny, nx][1]]) < np.linalg.norm([ny - self.obst_map[y, x][0], nx - self.obst_map[y, x][1]]):
            #     self.voro_map[y, x] = True
            # if np.linalg.norm([ny - self.obst_map[y, x][0], nx - self.obst_map[y, x][1]]) < np.linalg.norm([y - self.obst_map[ny, nx][0], x - self.obst_map[ny, nx][1]]):
            #     self.voro_map[ny, nx] = True
    
    def get_8_neighbors(self,y,x):
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            xn = x + dx
            yn = y + dy
            if xn < 0 or yn < 0:
                continue
            if xn >= self.width or yn >= self.height:
                continue
            neighbors.append([yn, xn])
        return neighbors
    
    def get_4_diag_neighbors(self,y,x):
        neighbors = []
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            xn = x + dx
            yn = y + dy
            if xn < 0 or yn < 0:
                continue
            if xn >= self.width or yn >= self.height:
                continue
            neighbors.append([yn, xn])
        return neighbors
    
    def get_4_manhattan_neighbors(self,y,x):
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            xn = x + dx
            yn = y + dy
            if xn < 0 or yn < 0:
                continue
            if xn >= self.width or yn >= self.height:
                continue
            neighbors.append([yn, xn])
        return neighbors
    
    def is_occupied(self,y,x):
        return np.array_equal(self.obst_map[y, x], [y,x])
    
    def compute_static(self):
        """Computes the initial distance map and Voronoi diagram."""
        queue = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 1:  # Obstacle
                    self.dist_map[y, x] = 0
                    self.obst_map[y, x] = [y, x]
                    heapq.heappush(queue, (0, y, x))
        # self.display()
        
        while queue:
            dist, y, x = heapq.heappop(queue)
            for ny, nx in self.get_8_neighbors(y,x): 
                new_dist = np.linalg.norm([ny - self.obst_map[y, x][0], nx - self.obst_map[y, x][1]])
                if new_dist < self.dist_map[ny, nx]:
                    self.dist_map[ny, nx] = new_dist
                    self.obst_map[ny, nx] = self.obst_map[y, x]
                    heapq.heappush(queue, (new_dist, ny, nx))
                else: # checkVoro
                    self.check_voro(y,x,ny,nx)
    
    def update(self, changed_cells):
        """Incrementally updates the distance map and Voronoi diagram."""
        queue = []
        toRaise = np.zeros_like(self.grid, dtype=bool)      
        for y, x, new_state in changed_cells:
            if new_state == 1:  # Newly occupied -> lower
                self.dist_map[y, x] = 0
                self.obst_map[y, x] = [y, x]
                heapq.heappush(queue, (0, y, x))
            else:  # Newly freed -> raise
                self.dist_map[y, x] = np.inf
                self.obst_map[y, x] = [-1, -1]
                toRaise[y,x] = True
                heapq.heappush(queue, (0, y, x))
        
        while queue:
            dist, y, x = heapq.heappop(queue)            
            if toRaise[y,x]: # Raise
                for ny, nx in self.get_8_neighbors(y,x):  # 8-connected
                    if (self.obst_map[ny, nx][0] != -1) and (not toRaise[ny,nx]):
                        heapq.heappush(queue, (self.dist_map[ny, nx], ny, nx))
                        if not self.is_occupied(self.obst_map[ny,nx][0],self.obst_map[ny,nx][1]):
                            self.dist_map[ny, nx] = np.inf
                            self.obst_map[ny, nx] = [-1, -1]
                            toRaise[ny,nx] = True
                toRaise[y,x] = False
            elif self.is_occupied(self.obst_map[y,x][0],self.obst_map[y,x][1]): # Lower
                self.voro_map[y, x] = False
                for ny, nx in self.get_8_neighbors(y,x):  # 8-connected
                    if not toRaise[ny,nx]:
                        new_dist = np.linalg.norm([self.obst_map[y, x][0]-ny, self.obst_map[y, x][1]-nx])
                        if new_dist < self.dist_map[ny, nx]:
                            self.dist_map[ny, nx] = new_dist
                            self.obst_map[ny, nx] = self.obst_map[y, x]
                            heapq.heappush(queue, (new_dist, ny, nx))
                        else: # checkVoro
                            self.check_voro(y,x,ny,nx)
    
    def combine_distance_maps(self, other):
        """Combines two distance maps by taking the minimum distance."""
        cells_to_update = []
        for y in range(self.height):
            for x in range(self.width):
                if other.dist_map[y,x] > 0 and self.dist_map[y,x] == 0:
                    cells_to_update.append((y,x,0))
        self.update(cells_to_update)
        
    def prune_4connected(self):
        queue = []
        for y in range(self.voro_map.shape[0]):
            for x in range(self.voro_map.shape[1]):
                ## Phase 1
                if not self.voro_map[y,x]:
                    neighbor_sum = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_4_manhattan_neighbors(y,x)])
                    if neighbor_sum == 4:
                        self.voro_map[y,x] = True
                
                if self.voro_map[y,x]:
                    heapq.heappush(queue, (self.dist_map[y,x],y,x))
          
        ## Phase 2: Keep only those that match the templates
        templates_2d = [
            # P^4_1
            [(0,-1,True),(-1,0,True),(-1,-1,False)],
            [(0,-1,True),(1,0,True),(1,-1,False)],
            [(0,1,True),(1,0,True),(1,1,False)],
            [(0,1,True),(-1,0,True),(-1,1,False)],
            # P^4_2
            [(0,-1,True),(1,0,False),(0,1,True),(-1,0,False)],
            [(0,-1,False),(1,0,True),(0,1,False),(-1,0,True)],
        ]
        while queue:
            dist, y, x = heapq.heappop(queue)
            matches_a_2d_template = False
            for template in templates_2d:
                matches_template = True
                for dy,dx,bool_val in template:
                    ty = y + dy
                    tx = x + dx
                    if 0 <= ty < self.height and 0 <= tx < self.width:
                        if self.voro_map[ty,tx] != bool_val:
                            matches_template = False
                            break
                    else:
                        matches_template = False
                        break
                if matches_template:
                    matches_a_2d_template = True
                    break
            if not matches_a_2d_template:
                self.voro_map[y,x] = False
    
    def prune_4connectedandkeep3neighbors(self):
        queue = []
        for y in range(self.voro_map.shape[0]):
            for x in range(self.voro_map.shape[1]):
                ## Phase 1
                if not self.voro_map[y,x]:
                    neighbor_sum = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_4_manhattan_neighbors(y,x)])
                    if neighbor_sum == 4:
                        self.voro_map[y,x] = True
                
                if self.voro_map[y,x]:
                    heapq.heappush(queue, (self.dist_map[y,x],y,x))
          
        ## Phase 2: Keep only those that match the templates
        templates_2d = [
            # P^4_1
            [(0,-1,True),(-1,0,True),(-1,-1,False)],
            [(0,-1,True),(1,0,True),(1,-1,False)],
            [(0,1,True),(1,0,True),(1,1,False)],
            [(0,1,True),(-1,0,True),(-1,1,False)],
            # P^4_2
            [(0,-1,True),(1,0,False),(0,1,True),(-1,0,False)],
            [(0,-1,False),(1,0,True),(0,1,False),(-1,0,True)],
        ]
        while queue:
            dist, y, x = heapq.heappop(queue)
            matches_a_2d_template = False
            for template in templates_2d:
                matches_template = True
                for dy,dx,bool_val in template:
                    ty = y + dy
                    tx = x + dx
                    if 0 <= ty < self.height and 0 <= tx < self.width:
                        if self.voro_map[ty,tx] != bool_val:
                            matches_template = False
                            break
                    else:
                        matches_template = False
                        break
                if matches_template:
                    matches_a_2d_template = True
                    break
            if not matches_a_2d_template:
                neighbor_sum = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_8_neighbors(y,x)])
                if neighbor_sum >= 4:
                    self.voro_map[y,x] = False
    
    def prune_8connected(self):
        queue = []
        for y in range(self.voro_map.shape[0]):
            for x in range(self.voro_map.shape[1]):
                ## Phase 1
                if not self.voro_map[y,x]:
                    neighbor_sum = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_4_manhattan_neighbors(y,x)])
                    if neighbor_sum == 4:
                        self.voro_map[y,x] = True
                
                if self.voro_map[y,x]:
                    heapq.heappush(queue, (self.dist_map[y,x],y,x))
          
        ## Phase 2: Keep only those that match the templates
        templates_2d = [
            # P^8_2
            [(0,-1,True),(1,0,False),(0,1,True),(-1,0,False)],
            [(0,-1,False),(1,0,True),(0,1,False),(-1,0,True)],
            # P^8_1
            [(0,-1,False),(-1,0,False),(-1,-1,True)],
            [(0,-1,False),(1,0,False),(1,-1,True)],
            [(0,1,False),(1,0,False),(1,1,True)],
            [(0,1,False),(-1,0,False),(-1,1,True)],            
            # P^8_3
            [(0,-1,True),(1,0,True),(0,1,True),(-1,0,True)],
        ]
        while queue:
            dist, y, x = heapq.heappop(queue)
            matches_a_2d_template = False
            for template in templates_2d:
                matches_template = True
                for dy,dx,bool_val in template:
                    ty = y + dy
                    tx = x + dx
                    if 0 <= ty < self.height and 0 <= tx < self.width:
                        if self.voro_map[ty,tx] != bool_val:
                            matches_template = False
                            break
                    else:
                        matches_template = False
                        break
                if matches_template:
                    matches_a_2d_template = True
                    break
            if not matches_a_2d_template:
                self.voro_map[y,x] = False
    
    def prune_4and8connected(self):
        queue = []
        for y in range(self.voro_map.shape[0]):
            for x in range(self.voro_map.shape[1]):
                ## Phase 1
                if not self.voro_map[y,x]:
                    neighbor_sum = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_4_manhattan_neighbors(y,x)])
                    if neighbor_sum == 4:
                        self.voro_map[y,x] = True
                
                if self.voro_map[y,x]:
                    heapq.heappush(queue, (self.dist_map[y,x],y,x))
          
        ## Phase 2: Keep only those that match the templates
        templates_2d = [
            # P^4_1
            [(0,-1,True),(-1,0,True),(-1,-1,False)],
            [(0,-1,True),(1,0,True),(1,-1,False)],
            [(0,1,True),(1,0,True),(1,1,False)],
            [(0,1,True),(-1,0,True),(-1,1,False)],
            # P^4_2 == P^8_2
            [(0,-1,True),(1,0,False),(0,1,True),(-1,0,False)],
            [(0,-1,False),(1,0,True),(0,1,False),(-1,0,True)],
            # P^8_1
            [(0,-1,False),(-1,0,False),(-1,-1,True)],
            [(0,-1,False),(1,0,False),(1,-1,True)],
            [(0,1,False),(1,0,False),(1,1,True)],
            [(0,1,False),(-1,0,False),(-1,1,True)],            
            # P^8_3
            [(0,-1,True),(1,0,True),(0,1,True),(-1,0,True)],
        ]
        while queue:
            dist, y, x = heapq.heappop(queue)
            matches_a_2d_template = False
            for template in templates_2d:
                matches_template = True
                for dy,dx,bool_val in template:
                    ty = y + dy
                    tx = x + dx
                    if 0 <= ty < self.height and 0 <= tx < self.width:
                        if self.voro_map[ty,tx] != bool_val:
                            matches_template = False
                            break
                    else:
                        matches_template = False
                        break
                if matches_template:
                    matches_a_2d_template = True
                    break
            if not matches_a_2d_template:
                self.voro_map[y,x] = False
    
    def prune_4and8connectedandkeep4plusneighbors(self):
        queue = []
        for y in range(self.voro_map.shape[0]):
            for x in range(self.voro_map.shape[1]):
                ## Phase 1
                if not self.voro_map[y,x]:
                    neighbor_sum = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_4_manhattan_neighbors(y,x)])
                    if neighbor_sum == 4:
                        self.voro_map[y,x] = True
                
                if self.voro_map[y,x]:
                    heapq.heappush(queue, (self.dist_map[y,x],y,x))
          
        ## Phase 2: Keep only those that match the templates
        templates_2d = [
            # P^4_1
            [(0,-1,True),(-1,0,True),(-1,-1,False)],
            [(0,-1,True),(1,0,True),(1,-1,False)],
            [(0,1,True),(1,0,True),(1,1,False)],
            [(0,1,True),(-1,0,True),(-1,1,False)],
            # P^4_2 == P^8_2
            [(0,-1,True),(1,0,False),(0,1,True),(-1,0,False)],
            [(0,-1,False),(1,0,True),(0,1,False),(-1,0,True)],
            # P^8_1
            [(0,-1,False),(-1,0,False),(-1,-1,True)],
            [(0,-1,False),(1,0,False),(1,-1,True)],
            [(0,1,False),(1,0,False),(1,1,True)],
            [(0,1,False),(-1,0,False),(-1,1,True)],            
            # P^8_3
            [(0,-1,True),(1,0,True),(0,1,True),(-1,0,True)],
        ]
        while queue:
            dist, y, x = heapq.heappop(queue)
            matches_a_2d_template = False
            for template in templates_2d:
                matches_template = True
                for dy,dx,bool_val in template:
                    ty = y + dy
                    tx = x + dx
                    if 0 <= ty < self.height and 0 <= tx < self.width:
                        if self.voro_map[ty,tx] != bool_val:
                            matches_template = False
                            break
                    else:
                        matches_template = False
                        break
                if matches_template:
                    matches_a_2d_template = True
                    break
            if not matches_a_2d_template:
                neighbor_sum = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_8_neighbors(y,x)])
                if neighbor_sum < 4:
                    self.voro_map[y,x] = False
    
    def point2line_distance(self, px, py, x1, y1, x2, y2):
        """Computes the perpendicular distance from a point (px, py) to a line 
        defined by points (x1, y1) and (x2, y2)."""
        num = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1))
        denom = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return num / denom if denom != 0 else float('inf')  # Avoid division by zero

    def max_deviation_point(self, edge_path):
        """Find the point in edge_path with the maximum deviation from the straight line."""
        p1, p2 = edge_path[0], edge_path[-1]
        max_dev = 0
        max_dev_point = None
        for pt in edge_path:
            if self.gvd_edges[pt]:
                dev = self.point2line_distance(pt[1],pt[0],p1[1],p1[0],p2[1],p2[0])
                if dev > max_dev:
                    max_dev = dev
                    max_dev_point = pt
        return max_dev_point, max_dev
    
    def mid_distance_point(self,edge_path):
        if not edge_path:
            return None, 0  # Return None if path is empty

        p1, p2 = np.array(edge_path[0]), np.array(edge_path[-1])
        line = p2-p1
        line_length = np.linalg.norm(line)
        line_midpt = line/2 + p1
        
        # Compute centroid (spatial middle)
        path_array = np.array(edge_path)

        # Find the point in edge_path closest to centroid
        distances = np.linalg.norm(path_array - line_midpt, axis=1)  # Euclidean distances
        min_idx = np.argmin(distances)

        return tuple(edge_path[min_idx]), line_length
    
    def floodfill_split_technique(self,max_iterations,max_dev_voxels,max_dist_voxels):
        # Initialize Nodes at Junctions + Corners, Edges = all else
        for y in range(self.voro_map.shape[0]):
            for x in range(self.voro_map.shape[1]):
                if not self.voro_map[y,x]:
                    continue
                # Don't add nodes next to each other
                is_neighbor_a_node = False
                for (ny,nx) in self.get_8_neighbors(y,x):
                    if self.gvd_nodes[ny,nx]:
                        is_neighbor_a_node = True
                        break
                if is_neighbor_a_node:
                    self.gvd_edges[y,x] = True
                    continue
                manhattan_neighbors = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_4_manhattan_neighbors(y,x)])
                diag_neighbors = sum([self.voro_map[ny,nx] for (ny,nx) in self.get_4_diag_neighbors(y,x)])
                if manhattan_neighbors >= 3 or diag_neighbors >= 3:
                    self.gvd_nodes[y,x] = True
                elif manhattan_neighbors == 1:#(diag_neighbors == 1 and manhattan_neighbors <= 1) or (manhattan_neighbors == 1 and diag_neighbors <= 1):
                    self.gvd_nodes[y,x] = True
                else:
                    self.gvd_edges[y,x] = True
        
        ## Iterate between 1) flood-fill and 2) split
        for iteration in range(max_iterations):
            ## 1) Flood-fill
            prev_nodes = {}
            self.gvd_ids = np.full_like(self.voro_map,-1,dtype=int)  # -1 means unlabeled
            node_id_cntr = 1
            dq = deque()
            for y in range(self.gvd_nodes.shape[0]):
                for x in range(self.gvd_nodes.shape[1]):
                    if self.gvd_nodes[y, x]:
                        self.gvd_ids[y, x] = node_id_cntr
                        prev_nodes[node_id_cntr] = (y,x)
                        dq.append((y, x, node_id_cntr))
                        node_id_cntr += 1
            while dq:
                y,x,node_id = dq.popleft()
                for (ny,nx) in self.get_4_manhattan_neighbors(y,x):
                    if self.gvd_edges[ny,nx] and self.gvd_ids[ny,nx] == -1:
                        self.gvd_ids[ny,nx] = node_id
                        dq.append((ny,nx,node_id))
            
            ## 2) Split: Identify connected nodes, split based on deviation or distance
            new_nodes = []
            connected_nodes = set()
            for y in range(self.gvd_ids.shape[0]):
                for x in range(self.gvd_ids.shape[1]):
                    if self.gvd_edges[y, x] and self.gvd_ids[y, x] != -1:
                        for (ny,nx) in self.get_4_manhattan_neighbors(y,x):
                            if self.gvd_edges[ny,nx] and self.gvd_ids[ny, nx] != -1 and self.gvd_ids[y, x] != self.gvd_ids[ny, nx]:
                                connected_nodes.add(tuple(sorted([(y, x), (ny, nx)])))
            for (n1, n2) in connected_nodes:
                nID1 = self.gvd_ids[n1[0],n1[1]]
                nID2 = self.gvd_ids[n2[0],n2[1]]
                edge_path = []
                queue = deque([n1])
                visited = set()
                num_end_nodes_found = 0
                while queue:
                    pt = queue.popleft()
                    if pt in visited:
                        continue
                    visited.add(pt)
                    edge_path.append(pt)
                    # Check if neighbor is a node, if so continue
                    neighbor_is_node = False
                    for (ny,nx) in self.get_4_manhattan_neighbors(pt[0],pt[1]):
                        if self.gvd_nodes[ny,nx]:
                            neighbor_is_node = True
                            num_end_nodes_found += 1
                            break
                    if num_end_nodes_found == 2:
                        break
                    elif neighbor_is_node:
                        continue
                    
                    for (ny,nx) in self.get_4_manhattan_neighbors(pt[0],pt[1]):
                        if (ny, nx) not in visited and self.gvd_edges[ny,nx]:
                            queue.append((ny, nx))
                
                # Add nodes to edge_path
                edge_path.insert(0,prev_nodes[nID1])
                edge_path.append(prev_nodes[nID2])
                
                if len(edge_path) > 4:
                    dev_split_point, deviation = self.max_deviation_point(edge_path)
                    dist_split_point, distance = self.mid_distance_point(edge_path)
                    if deviation > max_dev_voxels:
                        # print(f"Adding new node between {nID1} and {nID2} b/c of deviation: {deviation*0.2} [m]")
                        new_nodes.append(dev_split_point)
                    elif distance > max_dist_voxels: # If nodes are far enough apart, split halfway
                        # print(f"Adding new node between {nID1} and {nID2} b/c of distance: {distance*0.2} [m]")
                        new_nodes.append(dist_split_point)
            if not new_nodes:
                break  # No more splits needed, exit early
            for node in new_nodes:
                self.gvd_nodes[node] = True
                self.gvd_edges[node] = False
        
        ## Final Flood-Fill to update gvd_ids:
        self.gvd_ids = np.full_like(self.voro_map,-1,dtype=int)  # -1 means unlabeled
        node_id_cntr = 1
        dq = deque()
        for y in range(self.gvd_nodes.shape[0]):
            for x in range(self.gvd_nodes.shape[1]):
                if self.gvd_nodes[y, x]:
                    self.gvd_ids[y, x] = node_id_cntr
                    self.gvd_node_ids2coords[node_id_cntr] = tuple((y,x))
                    dq.append((y, x, node_id_cntr))
                    node_id_cntr += 1
        while dq:
            y,x,node_id = dq.popleft()
            for (ny,nx) in self.get_4_manhattan_neighbors(y,x):
                if self.gvd_edges[ny,nx] and self.gvd_ids[ny,nx] == -1:
                    self.gvd_ids[ny,nx] = node_id
                    dq.append((ny,nx,node_id))
        return

    def get_connected_node_ids(self):
        connected_node_ids = set()
        for y in range(self.gvd_ids.shape[0]):
            for x in range(self.gvd_ids.shape[1]):
                if self.gvd_edges[y, x] and self.gvd_ids[y, x] != -1:
                    for (ny,nx) in self.get_4_manhattan_neighbors(y,x):
                        if self.gvd_edges[ny,nx] and self.gvd_ids[ny, nx] != -1 and self.gvd_ids[y, x] != self.gvd_ids[ny, nx]:
                            connected_node_ids.add(tuple((self.gvd_ids[y, x], self.gvd_ids[ny, nx])))
        return connected_node_ids
    
    def get_connected_node_id_counts(self):
        connected_node_id_counts = {}
        for y in range(self.gvd_ids.shape[0]):
            for x in range(self.gvd_ids.shape[1]):
                if self.gvd_edges[y, x] and self.gvd_ids[y, x] != -1:
                    for (ny,nx) in self.get_4_manhattan_neighbors(y,x):
                        if self.gvd_edges[ny,nx] and self.gvd_ids[ny, nx] != -1 and self.gvd_ids[y, x] != self.gvd_ids[ny, nx]:
                            if self.gvd_ids[y, x] not in connected_node_id_counts:
                                connected_node_id_counts[self.gvd_ids[y, x]] = 1
                            else:
                                connected_node_id_counts[self.gvd_ids[y, x]] += 1
                            if self.gvd_ids[ny, nx] not in connected_node_id_counts:
                                connected_node_id_counts[self.gvd_ids[ny, nx]] = 1
                            else:
                                connected_node_id_counts[self.gvd_ids[ny, nx]] += 1
        return connected_node_id_counts
    
    def get_connected_node_coords(self):
        connected_node_ids = self.get_connected_node_ids()
        connected_nodes = []
        for (n1_id, n2_id) in connected_node_ids:
            connected_nodes.append(tuple((self.gvd_node_ids2coords[n1_id], self.gvd_node_ids2coords[n2_id])))
        return connected_nodes
    
    def display(self, plot_gvd=False, plot_node_edges=False, plot_flood_fill=False):
        """Displays the Voronoi diagram on top of the occupancy grid map."""
        fig, ax = plt.subplots()
        cmap = plt.cm.gray
        ax.imshow(self.dist_map, cmap=cmap, origin='upper')
        
        if plot_gvd:
            voro_overlay = np.zeros((*self.grid.shape, 4))
            voro_overlay[self.voro_map,:3] = [0, 1, 0]  # Green for Voronoi cells
            voro_overlay[self.voro_map, 3] = 0.5
            ax.imshow(voro_overlay, origin='upper')
        elif plot_node_edges:
            voro_overlay = np.zeros((*self.grid.shape, 4))
            voro_overlay[self.gvd_edges,:3] = [0,1,0]  # Green GVD Edges
            voro_overlay[self.gvd_edges, 3] = 0.5
            # voro_overlay[self.gvd_nodes,:3] = [1,0,0]  # Red GVD Nodes
            # voro_overlay[self.gvd_nodes, 3] = 0.5
            ax.imshow(voro_overlay, origin='upper')
            for y, x in zip(*np.where(self.gvd_nodes)):  # Extract coordinates where GVD is True
                circle = plt.Circle((x, y), 1, color=[1,0,0], alpha=1.0)
                ax.add_patch(circle)
        elif plot_flood_fill:
            unique_ids = np.unique(self.gvd_ids[self.gvd_ids != -1])  # Get unique node IDs
            num_colors = len(unique_ids)
            colors = plt.cm.get_cmap('prism',num_colors)  # Generate distinct colors
            edge_overlay = np.zeros((*self.grid.shape, 4))
            for i, node_id in enumerate(unique_ids):
                edge_overlay[self.gvd_ids == node_id,:3] = colors(i)[:3]  # Assign color
                edge_overlay[self.gvd_ids == node_id, 3] = 0.4
            ax.imshow(edge_overlay, origin='upper')
        # plt.show()
        return fig, ax
        

if __name__ == '__main__':
    # Example usage
    grid = np.zeros((20, 20), dtype=int)
    grid[5, 5] = grid[15, 15] = 1 # Obstacles
    # grid[70:75,40:42] = 1  # Obstacles
    dm = DistanceMap(grid)
    dm.compute_static()
    dm.display()
    dm.display(plot_gvd=True)

    # Simulating an update # state = 1 means add obstacle, and 0 means remove obstacle
    dm.update([(10, 10, 1), (5,15,1)])
    dm.display()
    dm.display(plot_gvd=True)
    
    dm.update([(10,10,0)])
    dm.display()
    dm.display(plot_gvd=True)