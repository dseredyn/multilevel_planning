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

class BehaviorKeepContact:
    def __init__(self):
        self.name = 'keep_contact'
        self.inputs = ['robot_pose', 'required_push', 'spatial_map']
        self.outputs = [ 'destination_geom', 'suppress_obstacle_avoidance', 'suppress_push' ]
        self.inputs_data = {}

        self.robot_dest_angle = None
        self.robot_dest_position = None

    def update(self, information):
        self.inputs_data = extractInformation( information, self.inputs )

        if not (self.inputs[0] in self.inputs_data and self.inputs[1] in self.inputs_data and self.inputs[2] in self.inputs_data):
            return []

        robot_pose = self.inputs_data['robot_pose']
        required_push = self.inputs_data['required_push']
        spatial_map = self.inputs_data['spatial_map']

        angle_diff = required_push.contact_force.get_angle_between( Vec2d(1,0).rotated(robot_pose.angle) )
        angle_diff = wrapAnglePI(angle_diff)
        torque = -angle_diff * 5.0
        if abs(torque) > 5:
            torque *= 5 / abs(torque)
#        torque = 20.0
#        print robot_pose.angle
        self.robot_dest_angle = robot_pose.angle - angle_diff

        robot_contact_w = Vec2d(-20.0, 0).rotated(self.robot_dest_angle) + robot_pose.position
        displacement = required_push.contact_point - robot_contact_w
        push_factor = (10.0 - displacement.get_length()) / 10.0
        if push_factor > 0.01:
            #print "BehaviorKeepContact: contact is OK"
            return [InformationSuppressObstacleAvoidance()]

        self.robot_dest_position = robot_pose.position + displacement

        # check if linear collision-less motion is possible
#        position = robot_pose.position
#        angle = robot_pose.angle

        self.robot_samples = [
            Vec2d(0, -40),
            Vec2d(15, -40),
            Vec2d(0, 40),
            Vec2d(15, 40),
            Vec2d(20, 0),
            Vec2d(-20, 0),
            Vec2d(0, 20),
            Vec2d(0, -20),
        ]

        obstacle_hit = False
        for t in np.linspace(0.0, 0.9, 30):
            position = t * self.robot_dest_position + (1.0-t) * robot_pose.position
            angle = t * self.robot_dest_angle + (1.0-t) * robot_pose.angle
            if array2Vec2d(position - self.robot_dest_position).get_length() < 10.0:
                break

            for pt in self.robot_samples:
                pt_W = pt.rotated(angle) + position
                s_idx = spatial_map.tri.find_simplex( np.asarray( [pt_W] ) )[0]
                if spatial_map.triangles_meaning[s_idx] == 0:
                    # obstacle
                    obstacle_hit = True
                    break

#        print "obstacle_hit", obstacle_hit
#        if not self.robot_dest_position is None:
#            pos_diff = (self.robot_dest_position - robot_pose.position)
#            dist = pos_diff.get_length()
#            if dist < 30.0:
#            pos_diff
#                return InformationRobotControl1(force, torque)

        #print "BehaviorKeepContact: change position"
        info_destination_geom = InformationDestinationGeom( self.robot_dest_position )
        info_destination_geom.angle = self.robot_dest_angle
        info_destination_geom.autoremove = True

        if obstacle_hit:
            #print "BehaviorKeepContact: go to pose"
            return [ info_destination_geom, InformationSuppressPush() ]
        else:
            #print "BehaviorKeepContact: suppress obstacle avoidance"
            return [ info_destination_geom, InformationSuppressObstacleAvoidance()]

    def debugVisDraw(self, debug_info):
        return
        if not self.robot_dest_position is None and not self.robot_dest_angle is None and not debug_info is None:
            p1 = self.robot_dest_position
            p2 = self.robot_dest_position + Vec2d(25.0, 0).rotated(self.robot_dest_angle)
            debug_info.append( ("circle", "green", p1, 20) )
            debug_info.append( ("vector", "green", p1, p2) )

