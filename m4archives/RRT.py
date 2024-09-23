from Obstacle import *
# from Practical04_Support.path_animation import *
# import meshcat.geometry as g
# import meshcat.transformations as tf

# from ece4078.Utility import StartMeshcat

import numpy as np
import random
import os
import types
import math
import time

# Import dependencies and set random seed
seed_value = 5
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)


# This is an adapted version of the RRT implementation done by Atsushi Sakai (@Atsushi_twi)
class RRTC:
    """
    Class for RRT planning
    """
    class Node:
        """
        RRT Node
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

        def __eq__(self, other):
            bool_list = []
            bool_list.append(self.x == other.x)
            bool_list.append(self.y == other.y)
            bool_list.append(np.all(np.isclose(self.path_x, other.path_x)))
            bool_list.append(np.all(np.isclose(self.path_y, other.path_y)))
            bool_list.append(self.parent == other.parent)
            return np.all(bool_list)

    def __init__(self, start=np.zeros(2),
                 goal=np.array([120,90]),
                 obstacle_list=None,
                 width = 160,
                 height=100,
                 expand_dis=3.0, 
                 path_resolution=0.5, 
                 max_points=200):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle_list: list of obstacle objects
        width, height: search area
        expand_dis: min distance between random node and closest node in rrt to it
        path_resolution: step size to considered when looking for node to expand
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.width = width
        self.height = height
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_nodes = max_points
        self.obstacle_list = obstacle_list
        self.start_node_list = [] # Tree from start
        self.end_node_list = [] # Tree from end

    def grow_tree(self, tree, node):
        # Extend a tree towards a specified node, starting from the closest node in the tree,
        # and return a Bolean specifying whether we should add the specified node or not
        # `added_new_node` is the Boolean.
        # If you plan on appending a new element to the tree, you should do that inside this function
        
        #TODO: Complete the method -------------------------------------------------------------------------
        # Extend the tree
        closest_ind = self.get_nearest_node_index(tree, node)
        closest_node = tree[closest_ind]
        
        # Check if we should add this node or not, and add it to the tree
        # 3. Select a node (nearby_node) close to expansion_node by moving from expantion_node to rnd_node
        # Use the steer method
        nearby_node = self.steer(closest_node, node, self.expand_dis)

        # 4. Check if nearby_node is in free space (i.e., it is collision free). If collision free, add node
        # to self.node_list
        
        # Check if the new node is collision-free
        if self.is_collision_free(nearby_node):
            tree.append(nearby_node)
            added_new_node = True

        # Not collision free
        else:
            added_new_node = False
        
        #ENDTODO ----------------------------------------------------------------------------------------------
        
        return added_new_node

    def check_trees_distance(self):
        # Find the distance between the trees, return if the trees distance is smaller than self.expand_dis
        # In other word, we are checking if we can connect the 2 trees.
        
        #TODO: Complete the method -------------------------------------------------------------------------
        
        # Default boolean state for can_be_connected
        can_be_connected = False

        # Iterate through nodes in the start tree
        for start_node in self.start_node_list:
            
            # Find the nearest node in the end tree to the current start node
            nearest_index = self.get_nearest_node_index(self.end_node_list, start_node)
            nearest_node = self.end_node_list[nearest_index]
            
            # Check if the distance between these nodes is within the expansion distance
            dist, _ = self.calc_distance_and_angle(start_node, nearest_node)

            # If the distance is within the allowable expansion distance
            if dist <= self.expand_dis:

                # Assign the variable to true
                can_be_connected = True
                break
        #ENDTODO ----------------------------------------------------------------------------------------------
        
        return can_be_connected

    def planning(self):
        """
        rrt path planning
        """
        self.start_node_list = [self.start]
        self.end_node_list = [self.end]

        # TODO
        print("Start node:", len(self.start_node_list))
        print("End node: :", len(self.end_node_list))
        print("Mac: ", self.max_nodes)
        count = 0
        while len(self.start_node_list) + len(self.end_node_list) <= self.max_nodes:
            
        #TODO: Complete the planning method ----------------------------------------------------------------
            # break # 0. Delete this line
            
            # 1. Sample and add a node in the start tree
            # Hint: You should use self.grow_tree above to add a node in the start tree here
            # count+= 1
            # print("Hi! Count:", count)
            rnd_node = self.get_random_node()
            print("Random Node: ", rnd_node)
            new_added_node = self.grow_tree(self.start_node_list, rnd_node)
            print("New Added Node:", new_added_node)
            print("Distance: ", self.check_trees_distance())
            
            # 2. Check whether trees can be connected
            # Hint: You should use self.check_trees_distance above to check.
            if new_added_node and self.check_trees_distance(): # If true, it will run the code below
                
            # 3. Add the node that connects the trees and generate the path

                print("I've runt hrough here")

                # This stops the expansion and generates the path after connecting them
                # Get the current start_node from the start_node_list
                current_start_node = self.start_node_list[-1]
                
                # Find the node in the end tree nearest to the current index of the start_node
                closest_ind = self.get_nearest_node_index(self.end_node_list, current_start_node)
                closest_node = self.end_node_list[closest_ind]
    
                # Finding a node which is even closer to try and connect the trees
                tree_connection_node = self.steer(closest_node, current_start_node, self.expand_dis)

                # Checking if this node is collision free
                if self.is_collision_free(tree_connection_node):

                    # If free, append and connect the two
                    self.end_node_list.append(tree_connection_node)
                    # Else, it will rerun the code after adding another end node onto the tree
                    
                    # Note: It is important that you return path found as:
                    return self.generate_final_course(len(self.start_node_list) - 1, len(self.end_node_list) - 1)
                    
            # 4. Sample and add a node in the end tree
            # Else, it is not close enough to connect them yet, so keep expanding
            rnd_node = self.get_random_node()
            self.grow_tree(self.end_node_list, rnd_node)
        
            # 5. Swap start and end trees
            self.start_node_list, self.end_node_list = self.end_node_list, self.start_node_list
            print("Start node list: ", self.start_node_list)
            print("End node list: ", self.end_node_list)
            print("\n\n\n\n\n------------------------------------")
        #ENDTODO ----------------------------------------------------------------------------------------------
            
        # self.node_list = [self.start]
        # while len(self.node_list) <= self.max_nodes:
            
        #     # 1. Generate a random node           
        #     rnd_node = self.get_random_node()
            
        #     # 2. Find node in tree that is closest to sampled node.
        #     # This is the node to be expanded (q_expansion)
        #     expansion_ind = self.get_nearest_node_index(self.node_list, rnd_node)
        #     expansion_node = self.node_list[expansion_ind]

        #     #TODO:  Complete the last two main steps of the RRT algorithm ----------------
        #     # 3. Select a node (nearby_node) close to expansion_node by moving from expantion_node to rnd_node
        #     # Use the steer method
        #     nearby_node = self.steer(expansion_node, rnd_node, self.expand_dis)
            
        #     # 4. Check if nearby_node is in free space (i.e., it is collision free). If collision free, add node
        #     # to self.node_list
        #     if self.is_collision_free(nearby_node):
        #         self.node_list.append(nearby_node)
            
        #     # Please remove return None when you start coding
        #     # return None
        #     #ENDTODO -----------------------------------------------------------------------
                
        #     # If we are close to goal, stop expansion and generate path
        #     if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
        #         final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
        #         if self.is_collision_free(final_node):
        #             return self.generate_final_course(len(self.node_list) - 1)
        #     print(self.node_list)
        # return None  # cannot find path
        
        
        # return None  # cannot find path
    
    # ------------------------------DO NOT change helper methods below ----------------------------
    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Given two nodes from_node, to_node, this method returns a node new_node such that new_node 
        is “closer” to to_node than from_node is.
        """
        
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        # How many intermediate positions are considered between from_node and to_node
        n_expand = math.floor(extend_length / self.path_resolution)

        # Compute all intermediate positions
        for _ in range(n_expand):
            new_node.x += self.path_resolution * cos_theta
            new_node.y += self.path_resolution * sin_theta
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)

        new_node.parent = from_node

        return new_node

    def is_collision_free(self, new_node):
        """
        Determine if nearby_node (new_node) is in the collision-free space.
        """
        if new_node is None:
            return True
        
        points = np.vstack((new_node.path_x, new_node.path_y)).T
        print("points: ", points)
        for obs in self.obstacle_list:
            in_collision = obs.is_in_collision_with_points(points)
            print("In collision result: ", in_collision)
            if in_collision:
                return False
        
        time.sleep(2)
        
        print("NO COLLISION")
        return True  # safe
    
    def generate_final_course(self, start_mid_point, end_mid_point):
        """
        Reconstruct path from start to end node
        """
        # First half
        node = self.start_node_list[start_mid_point]
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        
        # Other half
        node = self.end_node_list[end_mid_point]
        path = path[::-1]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        print("Width: ", self.width)
        print("Height: ", self.height)
        sample = np.random.random_sample() * 24 - 12
        x = self.width * (np.random.random_sample() * 24 - 12)
        y = self.height * (np.random.random_sample() * 24 - 12)
        print("X\t:", x, "\nY\t:", y)
        rnd = self.Node(x, y)
        print("Node!")
        time.sleep(1)
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):        
        # Compute Euclidean disteance between rnd_node and all nodes in tree
        # Return index of closest element
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y)
                 ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta        