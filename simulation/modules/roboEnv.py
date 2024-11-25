import time
import numpy as np
import pickle
import sys
import os
import pybullet as p
from init import *
from utils.tools import *
from modules.RRTstar import *
import pybullet_data
from modules.simple_control import *
from modules.dwa import *
import copy


class RobotEnv:
    def __init__(self, robotId, static_obstacles, dynamic_obstacle, target_position, available_position):
        self.robotId = robotId
        self.static_obstacles = static_obstacles
        self.dynamic_obstacle = dynamic_obstacle
        self.target_position = target_position
        target_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1])
        target_body_id = p.createMultiBody(baseVisualShapeIndex=target_visual_shape_id, basePosition=self.target_position)
        self.state = self.get_state()
        self.visited_pos = set()
        self.available_position = available_position

    def reset(self, reset_robo=True, reset_target=False):
        if reset_robo:
            reset_robot(p, self.robotId, [2,3,6,7])
        self.visited_pos = set()
        return self.get_state()
    
    def get_state(self):
        robo_pos = self.get_discrete_pos(self.robotId)
        dy_obs_pos = self.get_discrete_pos(self.dynamic_obstacle)
        target_pos = (round(self.target_position[0] / 0.25), round(self.target_position[1] / 0.25))
        return np.array([robo_pos[0], robo_pos[1], dy_obs_pos[0], dy_obs_pos[1], target_pos[0], target_pos[1], target_pos[0] - robo_pos[0], target_pos[1]-robo_pos[1]])

    def get_action_dir(self, action_idx):
        action_dir = [0,0]
        action_dir[0] = int(action_idx % 3 - 1)
        action_dir[1] = -int(action_idx / 3) + 1
        return tuple(action_dir)
    
    def get_discrete_pos(self, robotId):
        pos, ori = p.getBasePositionAndOrientation(robotId)
        discret_pos = (round(pos[0] / 0.25), round(pos[1] / 0.25))
        return discret_pos

    def get_real_pos(self, discrete_pos):
        return discrete_pos[0] * 0.25, discrete_pos[1] * 0.25

    def get_next_pos(self, x, y, action_dir):
        next_pos = (x + action_dir[0], y + action_dir[1])
        return next_pos

    def step(self, action_idx):
        state = self.get_state()
        action_dir = self.get_action_dir(action_idx)
        next_pos = self.get_next_pos(state[0], state[1], action_dir)
        if next_pos in self.available_position:
            p.resetBasePositionAndOrientation(self.robotId, [next_pos[0]*0.25, next_pos[1]*0.25,0.5], p.getQuaternionFromEuler([0, 0, 0]))
            for _ in range(5):
                p.stepSimulation()
        robo_pos = self.get_discrete_pos(self.robotId)
        target_pos = (round(self.target_position[0] / 0.25), round(self.target_position[1] / 0.25))
        self.visited_pos.add(self.get_discrete_pos(self.robotId))
        done = robo_pos == target_pos
        next_state = self.get_state()
        reward = self.get_reward(state, action_idx)
        return next_state, reward, done
    
    def check_collision(self, x0, y0, x, y):
        resolution = 0.01
        clearance = 0.15
        start = self.get_real_pos((x0, y0))
        end = self.get_real_pos((x, y))
        dist = np.hypot(start[0] - end[0], start[1] - end[1])
        steps = int(dist / resolution)
        obstacle_aabbs = [p.getAABB(obstacle_id) for obstacle_id in self.static_obstacles]
        dy_obstacle_aabb = p.getAABB(self.dynamic_obstacle)
        obstacle_aabbs.append(dy_obstacle_aabb)
        for i in range(steps + 1):
            x = start[0] + (end[0] - start[0]) * i / steps
            y = start[1] + (end[1] - start[1]) * i / steps
            for aabb_min, aabb_max in obstacle_aabbs:
                if (aabb_min[0]-clearance <= x <= aabb_max[0]+clearance) and (aabb_min[1]-clearance <= y <= aabb_max[1]+clearance):
                    return True
        return False

    def get_reward(self, last_state, action_idx):
        # compute collision
        x0, y0 = last_state[0], last_state[1]
        action_dir = self.get_action_dir(action_idx)
        npos = self.get_next_pos(x0, y0, action_dir)
        collision = False
        if npos not in self.available_position:
            collision = True

        # compute target dis
        new_state = self.get_state()
        x, y = new_state[0], new_state[1]
        target_pos = (round(self.target_position[0] / 0.25), round(self.target_position[1] / 0.25))
        target_relate_dis0 = np.hypot(target_pos[0] - x0, target_pos[1] - y0)
        target_relate_dis = np.hypot(target_pos[0] - x, target_pos[1] - y)
        
        speed = action_idx != 4
        explore_reward = 1 if (x,y) not in self.visited_pos else -0.1
        approaching_reward = target_relate_dis0 - target_relate_dis
        done = False
        if target_relate_dis < 0.2:
            done = True
    
        reward = - target_relate_dis*500 + explore_reward * 10 + collision*(-100) + speed*10 + done * 50000 + approaching_reward * 100
        return reward
    



class RobotEnvTest:
    def __init__(self, robotId, static_obstacles, dynamic_obstacle, target_position, available_position):
        self.robotId = robotId
        self.static_obstacles = static_obstacles
        self.dynamic_obstacle = dynamic_obstacle
        self.target_position = target_position
        target_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1])
        target_body_id = p.createMultiBody(baseVisualShapeIndex=target_visual_shape_id, basePosition=self.target_position)
        self.state = self.get_state()
        self.visited_pos = set()
        self.available_position = available_position

    def reset(self, reset_robo=True, reset_target=False):
        if reset_robo:
            reset_robot(p, self.robotId, [2,3,6,7])
        self.visited_pos = set()
        return self.get_state()
    
    def get_state(self):
        robo_pos = self.get_discrete_pos(self.robotId)
        dy_obs_pos = self.get_discrete_pos(self.dynamic_obstacle)
        target_pos = (round(self.target_position[0] / 0.25), round(self.target_position[1] / 0.25))
        return np.array([robo_pos[0], robo_pos[1], dy_obs_pos[0], dy_obs_pos[1], target_pos[0], target_pos[1], target_pos[0] - robo_pos[0], target_pos[1]-robo_pos[1]])

    def get_action_dir(self, action_idx):
        action_dir = [0,0]
        action_dir[0] = int(action_idx % 3 - 1)
        action_dir[1] = -int(action_idx / 3) + 1
        return tuple(action_dir)
    
    def get_discrete_pos(self, robotId):
        pos, ori = p.getBasePositionAndOrientation(robotId)
        discret_pos = (round(pos[0] / 0.25), round(pos[1] / 0.25))
        return discret_pos

    def get_real_pos(self, discrete_pos):
        return discrete_pos[0] * 0.25, discrete_pos[1] * 0.25

    def get_next_pos(self, x, y, action_dir):
        next_pos = (x + action_dir[0], y + action_dir[1])
        return next_pos

    def step(self, action_idx):
        state = self.get_state()
        action_dir = self.get_action_dir(action_idx)
        next_pos = self.get_next_pos(state[0], state[1], action_dir)
        if next_pos in self.available_position:
            simple_from_to_control(self.robotId, (next_pos[0]*0.25, next_pos[1]*0.25))
        robo_pos = self.get_discrete_pos(self.robotId)
        target_pos = (round(self.target_position[0] / 0.25), round(self.target_position[1] / 0.25))
        self.visited_pos.add(self.get_discrete_pos(self.robotId))
        done = robo_pos == target_pos
        next_state = self.get_state()
        reward = self.get_reward(state, action_idx)
        return next_state, reward, done
    
    def check_collision(self, x0, y0, x, y):
        resolution = 0.01
        clearance = 0.15
        start = self.get_real_pos((x0, y0))
        end = self.get_real_pos((x, y))
        dist = np.hypot(start[0] - end[0], start[1] - end[1])
        steps = int(dist / resolution)
        obstacle_aabbs = [p.getAABB(obstacle_id) for obstacle_id in self.static_obstacles]
        dy_obstacle_aabb = p.getAABB(self.dynamic_obstacle)
        obstacle_aabbs.append(dy_obstacle_aabb)
        for i in range(steps + 1):
            x = start[0] + (end[0] - start[0]) * i / steps
            y = start[1] + (end[1] - start[1]) * i / steps
            for aabb_min, aabb_max in obstacle_aabbs:
                if (aabb_min[0]-clearance <= x <= aabb_max[0]+clearance) and (aabb_min[1]-clearance <= y <= aabb_max[1]+clearance):
                    return True
        return False

    def get_reward(self, last_state, action_idx):
        # compute collision
        x0, y0 = last_state[0], last_state[1]
        action_dir = self.get_action_dir(action_idx)
        npos = self.get_next_pos(x0, y0, action_dir)
        collision = False
        if npos not in self.available_position:
            collision = True

        # compute target dis
        new_state = self.get_state()
        x, y = new_state[0], new_state[1]
        target_pos = (round(self.target_position[0] / 0.25), round(self.target_position[1] / 0.25))
        target_relate_dis0 = np.hypot(target_pos[0] - x0, target_pos[1] - y0)
        target_relate_dis = np.hypot(target_pos[0] - x, target_pos[1] - y)
        
        speed = action_idx != 4
        explore_reward = 1 if (x,y) not in self.visited_pos else -0.1
        approaching_reward = target_relate_dis0 - target_relate_dis
        done = False
        if target_relate_dis < 0.2:
            done = True
    
        reward = - target_relate_dis*500 + explore_reward * 10 + collision*(-100) + speed*10 + done * 50000 + approaching_reward * 100
        return reward