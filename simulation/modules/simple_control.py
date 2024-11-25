import numpy as np
import pybullet as p
from init import *
from utils.tools import *

pi = 3.1415926

def calculate_wheel_speeds(v, omega, wheel_base=1.08,  wheel_radius=0.036):
    v_left = (v - omega * wheel_base / 2) / wheel_radius
    v_right = (v + omega * wheel_base / 2) / wheel_radius
    return v_left, v_right


def get_relative_theta(relate_dis):
    if np.hypot(relate_dis[0], relate_dis[1]) > 0:
        target_theta = np.arcsin(relate_dis[0]/np.hypot(relate_dis[0], relate_dis[1]))
        if np.arcsin(relate_dis[1]/np.hypot(relate_dis[0], relate_dis[1])) > 0:
            target_theta = 3.1415926 - target_theta if target_theta > 0 else -3.1415926 - target_theta
    else:
        target_theta = 0
    return target_theta


def get_diff_theta(theta1, theta2):
    if theta2 - theta1 > 0:
        if theta2 - theta1 < pi:
            return theta2 - theta1
        else:
            return  -2*pi + theta2 - theta1
    else:
        if theta1 - theta2 < pi:
            return theta2 - theta1
        else:
            return 2*pi + theta2 - theta1


def simple_control(robotId, target_v, target_w, t, left_wheels=[6,7], right_wheels=[2,3]):
    v_left, v_right = calculate_wheel_speeds(target_v, target_w, wheel_base=0.54, wheel_radius=0.018)
    for i in range(int(t*240)):
        for joint in left_wheels:
            p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=v_left)
        for joint in right_wheels:
            p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=v_right)
        p.stepSimulation()
        

def simple_from_to_control(robotId, target, timestep=0.01):
    pi=3.1415926
    start, orientation = p.getBasePositionAndOrientation(robotId)
    _, _, theta = p.getEulerFromQuaternion(orientation)
    relate_dis = np.array(target) - np.array(start[:2])
    target_theta = get_relative_theta(relate_dis)
    w = 0.31415
    diff_theta = get_diff_theta(theta, target_theta)
    if diff_theta > 0:
        w = -w
    while np.hypot(relate_dis[0], relate_dis[1]) > 0.1:
        while abs(diff_theta) > 0.01:
            simple_control(robotId, target_v=0, target_w=w, t=timestep)
            start, orientation = p.getBasePositionAndOrientation(robotId)
            _, _, theta = p.getEulerFromQuaternion(orientation)
            relate_dis = np.array(target) - np.array(start[:2])
            target_theta = get_relative_theta(relate_dis)
            diff_theta = get_diff_theta(theta, target_theta)
        while abs(diff_theta) < 0.01 and np.hypot(relate_dis[0], relate_dis[1]) > 0.1:
            simple_control(robotId, target_v=0.1, target_w=0, t=timestep)
            start, orientation = p.getBasePositionAndOrientation(robotId)
            _, _, theta = p.getEulerFromQuaternion(orientation)
            relate_dis = np.array(target) - np.array(start[:2])


def simple_control_step(robotId, target_v, target_w, left_wheels=[6,7], right_wheels=[2,3]):
    v_left, v_right = calculate_wheel_speeds(target_v, target_w, wheel_base=0.54, wheel_radius=0.018)
    for joint in left_wheels:
        p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=v_left)
    for joint in right_wheels:
        p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=v_right)


def simple_from_to_control_step(robotId, target):
    pi=3.1415926
    start, orientation = p.getBasePositionAndOrientation(robotId)
    _, _, theta = p.getEulerFromQuaternion(orientation)
    relate_dis = np.array(target) - np.array(start[:2])
    target_theta = get_relative_theta(relate_dis)
    w = 0.31415
    diff_theta = get_diff_theta(theta, target_theta)
    if diff_theta > 0:
        w = -w
    if np.hypot(relate_dis[0], relate_dis[1]) > 0.2:
        if abs(diff_theta) > 0.01:
            simple_control_step(robotId, target_v=0, target_w=w*3)
        else: 
            simple_control_step(robotId, target_v=0.3, target_w=0)
    else:
        simple_control_step(robotId, target_v=0, target_w=0)


def dynamic_obstacle_step(obstacle, target, speed):
    pos, _ = p.getBasePositionAndOrientation(obstacle)
    relate_pos = (target[0] - pos[0], target[1] - pos[1])
    dis = np.hypot(relate_pos[0], relate_pos[1])
    if dis < 0.01:
        return True
    new_pos = [0,0, 0]
    new_pos[0] = pos[0] + speed * relate_pos[0] / dis
    new_pos[1] = pos[1] + speed * relate_pos[1] / dis
    p.resetBasePositionAndOrientation(obstacle, new_pos, p.getQuaternionFromEuler([0, 0, 0]))
    return False


def dynamic_obstacle_cycle(obstacle, path, idx, speed):
    reached = dynamic_obstacle_step(obstacle, path[idx], speed)
    if reached:
        return (idx + 1) % len(path)
    else:
        return idx