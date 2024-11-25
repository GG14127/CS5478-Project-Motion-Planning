import numpy as np
import pybullet as p
from init import *
from utils.tools import *
from modules.RRTstar import *
from modules.simple_control import *

class DynamicWindowApproach:
    def __init__(self, obstacle_list, robot_radius=0.2, max_v=5.0, max_w=3.0, v_res=0.1, w_res=0.1,
                 max_acc_v=2.5, max_acc_w=3.0, dt=0.1, goal_thresh=0.2):
        self.robot_radius = robot_radius
        self.max_v = max_v
        self.max_w = max_w
        self.v_res = v_res  # resolution of v
        self.w_res = w_res  # resolution of w
        self.max_acc_v = max_acc_v  # maximum acceleration of v
        self.max_acc_w = max_acc_w  # maximum acceleration of w
        self.dt = dt  # time step
        self.goal_thresh = goal_thresh
        self.obstacle_list = obstacle_list
        self.obstacle_aabbs = [p.getAABB(obstacle_id) for obstacle_id in obstacle_list]

    def calculate_dynamic_window(self, current_v, current_w):
        """calculate dynamic window for velocity sampling"""
        v_min = max(current_v - self.max_acc_v * self.dt, 0)
        v_max = min(current_v + self.max_acc_v * self.dt, self.max_v)
        w_min = max(current_w - self.max_acc_w * self.dt, -self.max_w)
        w_max = min(current_w + self.max_acc_w * self.dt, self.max_w)
        return v_min, v_max, w_min, w_max

    
    def next_pos(self, v, w, x, y, theta):
        traj = [(x, y, theta)]
        pi = 3.1415926
        # next pos in next time step
        x += v * np.sin(theta) * self.dt
        y -= v * np.cos(theta) * self.dt
        theta -= w * self.dt
        while theta > pi:
            theta -= 2 * pi
        while theta < - pi:
            theta += 2 * pi
        traj.append((x, y, theta))
        return traj
    
    def evaluate_next_pos(self, traj, goal):
        """calculate score of next position"""
        # score to the goal
        x0, y0, theta0 = traj[0]
        x, y, theta = traj[1]
        goal_cost = np.hypot(goal[0] - x, goal[1] - y)
        target_theta = get_relative_theta((goal[0] - x, goal[1] - y))
        theta_cost = abs(get_diff_theta(theta, target_theta))
        # collision and safety score
        collision = False
        safety_cost0 = float('inf')
        safety_cost = float('inf')
        obstacle_aabbs = [p.getAABB(obstacle_id) for obstacle_id in self.obstacle_list]
        for aabb_min, aabb_max in obstacle_aabbs:
            x0_dis = min(abs(x0 - aabb_min[0]), abs(x0 - aabb_max[0]))
            y0_dis = min(abs(y0 - aabb_min[1]), abs(y0 - aabb_max[1]))
            x_dis = min(abs(x - aabb_min[0]), abs(x - aabb_max[0]))
            y_dis = min(abs(y - aabb_min[1]), abs(y - aabb_max[1]))
            safety_cost0 = min(safety_cost0, np.hypot(x0_dis, y0_dis))
            safety_cost = min(safety_cost, np.hypot(x_dis, y_dis))
            
            if (aabb_min[0]-self.robot_radius <= x <= aabb_max[0]+self.robot_radius) and (aabb_min[1]-self.robot_radius <= y <= aabb_max[1]+self.robot_radius):
                collision = True
        # score for recover from collsion
        recover_cost = safety_cost - safety_cost0 if collision else 0
        # score for speed, encourage speed
        speed_cost = np.hypot(x-x0, y-y0) + abs(get_diff_theta(theta0, theta))
        return - goal_cost * 10 - theta_cost * 10 + safety_cost + speed_cost + recover_cost*100 + collision * (-1000)

    
    def plan(self, x, y, theta, current_v, current_w, goal):
        """plan the next step control velocity"""
        v_min, v_max, w_min, w_max = self.calculate_dynamic_window(current_v, current_w)
        best_score = -float('inf')
        best_v, best_w = 0, 0

        for v in np.arange(v_min, v_max, self.v_res):
            for w in np.arange(w_min, w_max, self.w_res):
                traj = self.next_pos(v, w, x, y, theta)
                score = self.evaluate_next_pos(traj, goal)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        return best_v, best_w