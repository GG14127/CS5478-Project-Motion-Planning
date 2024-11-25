import numpy as np
import pybullet as p
from init import *
from utils.tools import *

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.invalid = False 

class RRTStar:
    def __init__(self, start, goal, obstacle_list, bounds, step_size=0.3, search_radius=1.0, max_iter=10000):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacle_list = obstacle_list
        self.obstacle_aabbs = [p.getAABB(obstacle_id) for obstacle_id in obstacle_list]
        self.obstacle_aabbs.append(((0.2, -3.8, 0.0),(1.1, -2.7,1.)))
        self.obstacle_aabbs.append(((0.8, -4.2, 0.0),(1.2, -4.,1.)))
        self.obstacle_aabbs.append(((1.8, -3.8, 0.0),(2.5, -3.2,1.)))
        self.bounds = bounds
        self.step_size = step_size
        self.search_radius = search_radius
        self.max_iter = max_iter
        self.node_list = [self.start]

    def plan(self):
        for i in range(self.max_iter):
            rnd_node = self.get_random_node(i, self.max_iter)
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if not self.collision_check(new_node):
                continue

            self.node_list.append(new_node)
            self.rewire(new_node)

            # check if reached the goal
            if self.distance(new_node, self.goal) < self.step_size:
                self.goal.parent = new_node
                if self.collision_check(self.goal):
                    path = self.get_final_path(self.goal)
                    smoothed_path = self.get_smoothed_path(path)
                    return smoothed_path

        return None

    def get_random_node(self, iter_num, max_iter):
        """sample random nodes with goal oriented range"""
        if iter_num < max_iter * 0.8:  # global sampling
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            y = np.random.uniform(self.bounds[2], self.bounds[3])
        else:  # goal oriented sampling
            x = np.random.uniform(self.goal.x - 1, self.goal.x + 1)
            y = np.random.uniform(self.goal.y - 1, self.goal.y + 1)
        return Node(x, y)

    def get_nearest_node(self, rnd_node, k=5):
        """choose parent node for sampled node"""
        sorted_nodes = sorted(self.node_list, key=lambda node: self.distance(node, rnd_node))
        nearest_nodes = sorted_nodes[:k]
        return nearest_nodes[np.random.randint(len(nearest_nodes))]

    def steer(self, from_node, to_node):
        """generate new node by steering from the parent node to sampled node with step size"""
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        distance_to_goal = np.hypot(self.goal.x - to_node.x, self.goal.y - to_node.y)
        step_size = self.step_size
        new_x = from_node.x + step_size * np.cos(theta)
        new_y = from_node.y + step_size * np.sin(theta)
        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        return new_node

    def collision_check(self, node):
        """check collision from node to its parent"""
        if node.parent is None:
            return True  # start is always collision-free
        return self.segment_collision_check(node.parent, node, clearance=0.25)

    def segment_collision_check(self, from_node, to_node, resolution=0.01, clearance=0.25):
        """check path collision"""
        dist = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        steps = int(dist / resolution)
        for i in range(steps + 1):
            x = from_node.x + (to_node.x - from_node.x) * i / steps
            y = from_node.y + (to_node.y - from_node.y) * i / steps
            for aabb_min, aabb_max in self.obstacle_aabbs:
                if (aabb_min[0]-clearance <= x <= aabb_max[0]+clearance) and (aabb_min[1]-clearance <= y <= aabb_max[1]+clearance):
                    return False
        return True

    def rewire(self, new_node):
        """rewire the tree by evaluating cost to choose whether rewire the new node as parent"""
        for node in self.node_list:
            if self.distance(node, new_node) <= self.search_radius:
                if self.cost(new_node) + self.distance(node, new_node) < self.cost(node):
                    if self.segment_collision_check(new_node, node):
                        node.parent = new_node

    def cost(self, node):
        """compute cost from start to node"""
        cost = 0
        while node.parent is not None:
            cost += self.distance(node, node.parent)
            node = node.parent
        return cost

    def distance(self, node1, node2):
        """compute distance between nodes"""
        return np.hypot(node1.x - node2.x, node1.y - node2.y)

    def get_final_path(self, goal_node):
        """backtracking to find the final path"""
        path = []
        node = goal_node
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(self.start)
        return path[::-1]
    
    def get_smoothed_path(self, path):
        """smooth path by connecting to parent's parent if collision-free"""
        smoothed_path = []
        node = path[-1]
        smoothed_path.append((node.x, node.y))
        while node.parent:
            parent = node.parent
            while parent.parent and self.segment_collision_check(node, parent.parent):
                parent = parent.parent
            node.parent = parent
            node = node.parent
            smoothed_path.append((node.x, node.y))
        return smoothed_path[::-1]