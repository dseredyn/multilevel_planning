import sys
import math
import numpy as np
import random
import copy

import cv2
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.stats as stats

import pygame
from pygame.locals import *
from pygame.color import *
    
import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

from spatial_methods import *
from information import *

class BehaviorPushExecution:
    def __init__(self):
        self.name = 'push_execution'
        self.inputs = ['robot_pose', 'required_push', 'suppress_push']
        self.outputs = [ 'robot_control_4' ]
        self.inputs_data = {}

    def update(self, information):
        self.inputs_data = extractInformation( information, self.inputs )

        if not (self.inputs[0] in self.inputs_data and self.inputs[1] in self.inputs_data):
            return []

        if self.inputs[2] in self.inputs_data:
            suppress_push = self.inputs_data[self.inputs[2]]
            suppress_push.read = True
        else:
            suppress_push = None

        if not suppress_push is None:
            return []

        robot_pose = self.inputs_data['robot_pose']
        required_push = self.inputs_data['required_push']

        angle_diff = required_push.contact_force.get_angle_between( Vec2d(1,0).rotated(robot_pose.angle) )
        angle_diff = wrapAnglePI(angle_diff)
        torque = -angle_diff * 5.0
        if abs(torque) > 5:
            torque *= 5 / abs(torque)
#        torque = 20.0
#        print robot_pose.angle

        robot_contact_w = Vec2d(-20.0, 0).rotated(robot_pose.angle) + robot_pose.position
        displacement = required_push.contact_point - robot_contact_w
        push_factor = (10.0 - displacement.get_length()) / 10.0
        push_factor = min(1.0, max(0.0, push_factor))
        if push_factor < 0.01:
            return []
        force = displacement - required_push.contact_force*push_factor
        force *= 0.1
        force_mag = force.get_length()
        if force_mag > 1.0:
            force /= force_mag
        #force = Vec2d()
        #return []
        return [ InformationRobotControl4(force, torque) ]

    def debugVisDraw(self, debug_info):
        assert False
        if not self.best_surf is None and not debug_info is None:
            pt, n = self.best_surf
            n_w = n.rotated(self.obj.angle)
            pt_w = pt.rotated(self.obj.angle) + self.obj.position
            debug_info.append( ("vector", "blue", pt_w, pt_w + n_w * 50.0) )

            for surf in self.obj.surface:
                pt, n = surf
                p = pt.rotated(self.best_angle) + self.best_position
                debug_info.append( ("circle", "green", p, 2) )

