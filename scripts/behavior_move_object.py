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

class BehaviorMoveObject:
    def __init__(self):
        self.name = 'move_object_command'
        self.inputs = [ 'move_object_command' ]#, 'spatial_map', 'movement_prediction']
        self.outputs = [ 'expected_motion', 'focus' ]
        self.inputs_data = {}

        self.predicted_movement = None

    def update(self, information):
        self.inputs_data = extractInformation( information, self.inputs )

        if not (self.inputs[0] in self.inputs_data):# and self.inputs[1] in self.inputs_data):
            return []

        move_object_command = self.inputs_data[ self.inputs[0] ]

        # for now, we assume that there are no obstacles for door opening motion
        info_expected_motion = InformationExpectedMotion()
        info_expected_motion.anchor = move_object_command.object_anchor
        info_expected_motion.target_point = move_object_command.dest_position

        info_focus = InformationFocus()
        info_focus.object_name = move_object_command.object_anchor
        return [ info_expected_motion, info_focus ]

