#import sys
import math
#import numpy as np
#import random

#import cv2
#import matplotlib.pyplot as plt
#import scipy.spatial
#import scipy.stats as stats

import pygame
from pygame.locals import *
#from pygame.color import *
    
#import pymunk
from pymunk.vec2d import Vec2d
#import pymunk.pygame_util
from spatial_methods import *

def manualForceControl(robot, keys):
    if (keys[K_UP]):
        robot.apply_impulse_at_local_point(Vec2d(0,100), (0,0))
    if (keys[K_DOWN]):
        robot.apply_impulse_at_local_point(Vec2d(0,-100), (0,0))
    if (keys[K_LEFT]):
        robot.apply_impulse_at_local_point(Vec2d(-100,0), (0,0))
    if (keys[K_RIGHT]):
        robot.apply_impulse_at_local_point(Vec2d(100,0), (0,0))

class ManualPositionControl:
    def __init__(self, lin_stiffness, lin_damping, rot_stiffness, rot_damping):
        self.lin_stiffness = lin_stiffness
        self.lin_damping = lin_damping
        self.rot_stiffness = rot_stiffness
        self.rot_damping = rot_damping
        self.target = None

    def update(self, robot, keys):
        if self.target is None:
            self.target = [ robot.position[0], robot.position[1], robot.angle ]

        if (keys[K_UP]):
            self.target[0] += math.cos(self.target[2])
            self.target[1] += math.sin(self.target[2])
        if (keys[K_DOWN]):
            self.target[0] -= math.cos(self.target[2])
            self.target[1] -= math.sin(self.target[2])

        if (keys[K_s]):
            self.target[0] += math.sin(self.target[2])
            self.target[1] -= math.cos(self.target[2])
        if (keys[K_a]):
            self.target[0] -= math.sin(self.target[2])
            self.target[1] += math.cos(self.target[2])

        if (keys[K_LEFT]):
            self.target[2] += 0.05
        if (keys[K_RIGHT]):
            self.target[2] -= 0.05
        robot.force = self.lin_stiffness * (Vec2d(self.target[0], self.target[1]) - robot.position) - self.lin_damping * robot.velocity
        robot.torque = self.rot_stiffness * wrapAnglePI( self.target[2] - robot.angle ) - self.rot_damping * robot.angular_velocity

    def debugInfoDraw(self, debug_info=None):
        if not debug_info is None and not self.target is None:
            p1 = Vec2d(self.target[0], self.target[1])
            p2 = Vec2d(self.target[0] + 15*math.cos(self.target[2]), self.target[1] + 15*math.sin(self.target[2]))
            debug_info.append( ("circle", "green", p1, 10) )
            debug_info.append( ("vector", "green", p1, p2) )

