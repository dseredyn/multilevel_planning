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

# This behavior uses path, spatial_map, destination and current position of robot
# to generate information about movement of robot, i.e. virtual force and torque
# that pushes the robot in desired direction.
class BehaviorRobotControl:
    def __init__(self):
        self.name = 'robot_control'
        self.inputs = ['robot_control_1', 'robot_control_2', 'robot_control_3', 'robot_control_4']
        self.outputs = [ 'robot_total_control' ]

        self.info_robot_control_1 = None
        self.info_robot_control_2 = None
        self.info_robot_control_3 = None
        self.info_robot_control_4 = None

    def update(self, information):
        self.info_robot_control_1 = None
        self.info_robot_control_2 = None
        self.info_robot_control_3 = None
        self.info_robot_control_4 = None

        for info in information:
            if info.type == 'robot_control_1':
                self.info_robot_control_1 = info
                info.read = True
            elif info.type == 'robot_control_2':
                self.info_robot_control_2 = info
                info.read = True
            elif info.type == 'robot_control_3':
                self.info_robot_control_3 = info
                info.read = True
            elif info.type == 'robot_control_4':
                self.info_robot_control_4 = info
                info.read = True

        if self.info_robot_control_1 is None and self.info_robot_control_2 is None and self.info_robot_control_3 is None and self.info_robot_control_4 is None:
            self.valid = False
        else:
            self.valid = True

        if not self.valid:
            return []

        force = 0.0
        torque = 0.0

        if not self.info_robot_control_1 is None:
            force += self.info_robot_control_1.force
            torque += self.info_robot_control_1.torque

        if not self.info_robot_control_2 is None:
            force += self.info_robot_control_2.force
            torque += self.info_robot_control_2.torque

        if not self.info_robot_control_3 is None:
            force += self.info_robot_control_3.force
            torque += self.info_robot_control_3.torque

        if not self.info_robot_control_4 is None:
            force += self.info_robot_control_4.force
            torque += self.info_robot_control_4.torque

        return [ InformationRobotTotalControl(force, torque) ]

