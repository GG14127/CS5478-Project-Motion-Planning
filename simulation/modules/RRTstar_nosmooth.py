import numpy as np
import pybullet as p
from init import *
from utils.tools import *

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.invalid = False  # 标记无效节点

class RRTStar:
    def __init__(self, start, goal, obstacle_list, bounds, step_size=0.4, search_radius=1.0, max_iter=6000):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacle_list = obstacle_list
        self.obstacle_aabbs = [p.getAABB(obstacle_id) for obstacle_id in obstacle_list]
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

            # 检查是否到达目标
            if self.distance(new_node, self.goal) < self.step_size:
                self.goal.parent = new_node
                if self.collision_check(self.goal):
                    return self.get_final_path(self.goal)

        return None

    def get_random_node(self, iter_num, max_iter):
        """动态调整采样范围"""
        if iter_num < max_iter * 0.8:  # 前 70% 的迭代做全局探索
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            y = np.random.uniform(self.bounds[2], self.bounds[3])
        else:  # 后 30% 的迭代集中采样目标区域附近
            x = np.random.uniform(self.goal.x - 1, self.goal.x + 1)
            y = np.random.uniform(self.goal.y - 1, self.goal.y + 1)
        return Node(x, y)

    def get_nearest_node(self, rnd_node, k=5):
        """选择离采样点最近的 k 个节点，随机选一个作为 parent"""
        sorted_nodes = sorted(self.node_list, key=lambda node: self.distance(node, rnd_node))
        nearest_nodes = sorted_nodes[:k]
        return nearest_nodes[np.random.randint(len(nearest_nodes))]

    def steer(self, from_node, to_node):
        """基于目标引导 steer"""
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
        distance_to_goal = np.hypot(self.goal.x - to_node.x, self.goal.y - to_node.y)
        step_size = self.step_size
        new_x = from_node.x + step_size * np.cos(theta)
        new_y = from_node.y + step_size * np.sin(theta)
        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        return new_node

    def collision_check(self, node):
        """逐点检测路径段的碰撞"""
        if node.parent is None:
            return True  # 起始点总是有效
        return self.segment_collision_check(node.parent, node, clearance=0.25)

    def segment_collision_check(self, from_node, to_node, resolution=0.01, clearance=0.4):
        """逐点检测路径段是否碰撞"""
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
        """重连节点以优化路径"""
        for node in self.node_list:
            if self.distance(node, new_node) <= self.search_radius:
                if self.cost(new_node) + self.distance(node, new_node) < self.cost(node):
                    if self.segment_collision_check(new_node, node):
                        node.parent = new_node

    def cost(self, node):
        """计算到某节点的路径代价"""
        cost = 0
        while node.parent is not None:
            cost += self.distance(node, node.parent)
            node = node.parent
        return cost

    def distance(self, node1, node2):
        """计算两节点之间的距离"""
        return np.hypot(node1.x - node2.x, node1.y - node2.y)

    def get_final_path(self, goal_node):
        """回溯生成最终路径"""
        path = []
        node = goal_node
        while node.parent is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.append((self.start.x, self.start.y))
        return path[::-1]