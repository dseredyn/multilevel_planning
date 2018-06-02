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
class BehaviorObjectPerception:
    def __init__(self):
        self.name = 'object_perception'
        self.inputs = ['env_perception', 'focus']
        self.outputs = [ ]
        self.inputs_data = {}

    def update(self, information):
        self.inputs_data = extractInformation( information, self.inputs )

        if not (self.inputs[0] in self.inputs_data and self.inputs[1] in self.inputs_data):
            return []

        info_env = self.inputs_data[self.inputs[0]]
        info_focus = self.inputs_data[self.inputs[1]]

        for m in info_env.models:
            if m.name == info_focus.object_name:
                info_inst = InformationObjectInstance()
                info_inst.type = 'perception_' + m.name
                info_inst.position = m.body.position
                info_inst.angle = m.body.angle
                info_inst.velocity = m.body.velocity
                info_inst.angular_velocity = m.body.angular_velocity
                info_inst.mass = m.body.mass
                info_inst.moment = m.body.moment
                info_inst.surface = m.surface

                self.outputs = [ info_inst.type ]

                return [ info_inst ]
        return []

class BehaviorDoorPerception:
    def __init__(self):
        self.name = 'door_perception'
        self.inputs = ['env_perception', 'focus']
        self.outputs = [ ]
        self.inputs_data = {}

    def update(self, information):
        self.inputs_data = extractInformation( information, self.inputs )

        if not (self.inputs[0] in self.inputs_data and self.inputs[1] in self.inputs_data):
            return []

        info_env = self.inputs_data[self.inputs[0]]
        info_focus = self.inputs_data[self.inputs[1]]

        for m in info_env.models:
            if m.name == info_focus.object_name and m.type == 'door':
                info_inst = InformationObjectInstance()
                info_inst.type = 'perception_door_' + m.name
                info_inst.position = m.body.position
                info_inst.angle = m.body.angle
                info_inst.velocity = m.body.velocity
                info_inst.angular_velocity = m.body.angular_velocity
                info_inst.mass = m.body.mass
                info_inst.moment = m.body.moment
                info_inst.surface = m.surface

                info_inst.open_angle = m.getOpenAngle()
                info_inst.hinge_pos = m.body.position
                info_inst.left = m.left

                self.outputs = [ info_inst.type ]
                return [ info_inst ]
        return []

