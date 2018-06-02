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
class BehaviorOpenDoor:
    def __init__(self):
        self.name = 'open_door'
        self.inputs = ['open_door_command']
        self.outputs = [ 'expected_motion', 'focus' ]
        self.inputs_data = {}

        self.anchored_object_data_name = None

    def update(self, information):
        if not self.anchored_object_data_name is None and not self.anchored_object_data_name in self.inputs:
            self.inputs.append( self.anchored_object_data_name )
        self.inputs_data = extractInformation( information, self.inputs )

        if not (self.inputs[0] in self.inputs_data):# and self.inputs[1] in self.inputs_data and self.inputs[2] in self.inputs_data):
            return []

        open_door_command = self.inputs_data[ self.inputs[0] ]
        self.anchored_object_data_name = 'perception_door_' + open_door_command.object_anchor
        if not self.anchored_object_data_name in self.inputs_data:
            print "BehaviorOpenDoor: object not found:", open_door_command.object_anchor, "data name:", self.anchored_object_data_name
            info_focus = InformationFocus()
            info_focus.object_name = open_door_command.object_anchor
            return [ info_focus ]

        door_info = self.inputs_data[self.anchored_object_data_name]

        # generate destination pose for the door
        # door are open if their open_angle is over 80 degrees
        dest_open_angle = open_door_command.open_angle
        if abs(door_info.open_angle) > dest_open_angle:
            return []
            ## TODO: the task is completed - handle this case
            #raise Exception('not implemented')

        if door_info.left:
            # open_angle should be positive
            angle_diff = dest_open_angle - door_info.open_angle
        else:
            # open_angle should be negative
            angle_diff = -dest_open_angle - door_info.open_angle
        #print "angle_diff", angle_diff, dest_open_angle, door_info.open_angle

        if abs(angle_diff) < math.pi/10.0:
            #print "angle_diff", angle_diff
            return []
#        info_move = InformationMovement()
#        info_move.object_anchor = 'door_instance'
#        info_move.current_rotation = self.inputs_data['door_instance'].angle
#        info_move.current_position = self.inputs_data['door_instance'].hinge_pos
#        info_move.dest_rotation = self.inputs_data['door_instance'].angle + angle_diff
#        info_move.dest_position = self.inputs_data['door_instance'].hinge_pos

        dest_rotation = door_info.angle + angle_diff

        info_expected_motion = InformationExpectedMotion()
        info_expected_motion.anchor = open_door_command.object_anchor
        #info_expected_motion.target_point = door_info.position + Vec2d(100,0).rotated(dest_rotation)
        info_expected_motion.target_angle = dest_rotation
        info_expected_motion.autoremove = True

#        InformationMoveObjectCommand("box", Vec2d(500,500))

        info_focus = InformationFocus()
        info_focus.object_name = open_door_command.object_anchor

        return [ info_expected_motion, info_focus ]

