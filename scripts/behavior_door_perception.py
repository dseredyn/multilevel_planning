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

# This behavior is highly abstract. It interprets the command 'open door' and
# generates outputs:
# - movement information
class BehaviorDoorPerception:
    def __init__(self, door_object):
        self.name = 'door_perception'
        self.inputs = ['env_perception']
        self.outputs = [ 'cabinet_left_door' ]
        self.inputs_data = {}

        self.door_object = door_object

    def update(self, information):
        self.inputs_data = extractInformation( information, self.inputs )

        if self.inputs[0] in self.inputs_data:# and self.inputs[1] in self.inputs_data:
            # parse inputs
            pass

        info_door = InformationDoorInstance()
        info_door.type = "cabinet_left_door"
        info_door.mass = self.door_object.body.mass
        info_door.moment = self.door_object.body.moment
        info_door.open_angle = self.door_object.getOpenAngle()
        info_door.angle = self.door_object.body.angle
        info_door.position = self.door_object.body.position
        info_door.velocity = self.door_object.body.velocity
        info_door.angular_velocity = self.door_object.body.angular_velocity
        info_door.hinge_pos = self.door_object.body.position
        info_door.limits = self.door_object.limits
        info_door.length = self.door_object.length
        info_door.width = self.door_object.width
        info_door.left = self.door_object.left
        info_door.surface = self.door_object.surface

        #print info_door.hinge_pos
        return [ info_door ]

