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

def joinInformation(info_old, info_new):
    info_to_add = []
    for i_new in range(len(info_new)):
        found = False
        for i_old in range(len(info_old)):
            if info_new[i_new].type == info_old[i_old].type:
                info_old[i_old] = info_new[i_new]
                found = True
                break
        if not found:
            info_to_add.append( info_new[i_new] )

    for inf in info_to_add:
        info_old.append( inf )

def clearObsoleteInformation(info):
    for i in reversed(range(len(info))):
        if info[i].autoremove and info[i].read:
            info.remove(info[i])

def extractInformation(information, inputs):
    result = {}
    for info in information:
        for inp in inputs:
            if info.type == inp:
                result[inp] = info
    return result

class InformationSpatialMap:
    def __init__(self):
        self.type = "spatial_map"
        self.autoremove = False
        self.read = False

        self.edges_dict = {}
        self.edges_lengths = {}
        self.circumcenters = None
        self.meancenters = None
        self.circumradius = None
        #self.dist_map = None
        self.tri = None
        self.triangles_meaning = None

        self.spatial_map_id = None

class InformationSpatialMapRange:
    def __init__(self, range_x, steps_x, range_y, steps_y):
        self.type = "spatial_map_range"
        self.autoremove = False
        self.read = False

        self.range_x = range_x
        self.steps_x = steps_x
        self.range_y = range_y
        self.steps_y = steps_y

class InformationRobotPose:
    def __init__(self, position, angle):
        self.type = "robot_pose"
        self.autoremove = True
        self.read = False

        self.position = position
        self.angle = angle

class InformationPerceptionSpace:
    def __init__(self, space, models):
        self.type = "env_perception"
        self.autoremove = False
        self.read = False

        self.space = space
        self.models = models

class InformationDestinationGeom:
    def __init__(self, point):
        self.type = "destination_geom"
        self.autoremove = False
        self.read = False

        self.point = point
        self.angle = None

class InformationPathGeom:
    def __init__(self, path):
        self.type = "path"
        self.autoremove = False

        self.path = path
        self.prev = None

class InformationRobotControl1:
    def __init__(self, force, torque):
        self.type = "robot_control_1"
        self.autoremove = True
        self.read = False

        self.force = force
        self.torque = torque

class InformationRobotControl2:
    def __init__(self, force, torque):
        self.type = "robot_control_2"
        self.autoremove = True
        self.read = False

        self.force = force
        self.torque = torque

class InformationRobotControl3:
    def __init__(self, force, torque):
        self.type = "robot_control_3"
        self.autoremove = True
        self.read = False

        self.force = force
        self.torque = torque

class InformationRobotControl4:
    def __init__(self, force, torque):
        self.type = "robot_control_4"
        self.autoremove = True
        self.read = False

        self.force = force
        self.torque = torque

class InformationRobotTotalControl:
    def __init__(self, force, torque):
        self.type = "robot_total_control"
        self.autoremove = True
        self.read = False

        self.force = force
        self.torque = torque

class InformationFlow:
    def __init__(self, flow_vector):
        self.type = "flow_vector"
        self.autoremove = False
        self.read = False

        self.flow_vector = flow_vector

class InformationRobotHasStuck:
    def __init__(self):
        self.type = "stuck"
        self.autoremove = True
        self.read = False

class InformationDoorInstance:
    def __init__(self):
        self.type = "door_instance"
        self.autoremove = True
        self.read = False

class InformationOpenDoorCommand:
    def __init__(self, object_anchor, open_angle):
        self.type = "open_door_command"
        self.autoremove = False
        self.read = False

        self.object_anchor = object_anchor
        self.open_angle = open_angle

class InformationMovement:
    def __init__(self):
        self.type = "movement_information"
        self.autoremove = False
        self.read = False

        self.object_anchor = None
        self.current_rotation = None
        self.current_position = None
        self.dest_rotation = None
        self.dest_position = None

class InformationMoveObjectCommand:
    def __init__(self, object_anchor, dest_position):
        self.type = "move_object_command"
        self.autoremove = False
        self.read = False

        self.object_anchor = object_anchor
        #self.current_rotation = None
        #self.current_position = None
        #self.dest_rotation = None
        self.dest_position = dest_position

class InformationExpectedMotion:
    def __init__(self):
        self.type = "expected_motion"
        self.autoremove = False
        self.read = False

        self.anchor = None
        self.target_point = None
        self.target_angle = None

class InformationObjectInstance:
    def __init__(self):
        self.type = None
        self.autoremove = False
        self.read = False

        self.surface = None
        self.position = None
        self.angle = None
        self.mass = None
        self.moment = None
        self.velocity = None
        self.angular_velocity = None

class InformationRequiredPush:
    def __init__(self):
        self.type = "required_push"
        self.autoremove = False
        self.read = False

        self.anchor = None
        self.contact_point = None
        self.contact_normal = None
        self.contact_force = None

class InformationFocus:
    def __init__(self):
        self.type = "focus"
        self.autoremove = False
        self.read = False

        self.object_name = None

class InformationSuppressObstacleAvoidance:
    def __init__(self):
        self.type = "suppress_obstacle_avoidance"
        self.autoremove = True
        self.read = False

class InformationSuppressPush:
    def __init__(self):
        self.type = "suppress_push"
        self.autoremove = True
        self.read = False

