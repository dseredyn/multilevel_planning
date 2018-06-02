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
class BehaviorObstacleAvoidance:
    def __init__(self):
        self.name = 'obstacle_avoidance'
        self.inputs = ['robot_pose', 'spatial_map', 'flow_vector', 'suppress_obstacle_avoidance']
        self.outputs = [ 'robot_control_2' ]
        self.inputs_data = {}

        self.info_robot_pose = None
        self.info_spatial_map = None
        self.info_flow_vector = None

        self.rep_vec = None
        self.flow = None
        self.flow2 = None

    def update(self, information):
        self.inputs_data = extractInformation( information, self.inputs )

        self.info_robot_pose = None
        self.info_spatial_map = None
        self.info_flow_vector = None

        for info in information:
            if info.type == 'robot_pose':
                self.info_robot_pose = info
            elif info.type == 'spatial_map':
                self.info_spatial_map = info
            elif info.type == "flow_vector":
                self.info_flow_vector = info

        if not self.inputs[0] in self.inputs_data or not self.inputs[1] in self.inputs_data or not self.inputs[2] in self.inputs_data:
            self.valid = False
        else:
            self.valid = True

        if not self.valid:
            return []

        if self.inputs[3] in self.inputs_data:
            suppress_obstacle_avoidance = self.inputs_data[self.inputs[3]]
            suppress_obstacle_avoidance.read = True
        else:
            suppress_obstacle_avoidance = None

        if suppress_obstacle_avoidance is None:
            force_vec = self.getDrivingForce()
            if force_vec is None:
                return []

            return [ InformationRobotControl2(force_vec, 0.0) ]
        else:
            #print "BehaviorObstacleAvoidance: suppress"
            return []

    def getDrivingForce(self):
        current_pos = self.info_robot_pose.position
#        tri_vor = self.info_spatial_map.tri_vor
        tri = self.info_spatial_map.tri

#        s_idx = tri_vor.find_simplex( np.asarray( [current_pos] ) )[0]
#        s = tri_vor.simplices[s_idx,:]
#        num_border_points = self.info_spatial_map.tri.points.shape[0]
#        mean = Vec2d()
#        for i in range(3):
#            mean += array2Vec2d( tri_vor.points[s[i],:] )
#        mean /= 3.0

        self.flow = self.info_flow_vector.flow_vector
#        self.flow2 = self.info_flow_vector.flow_vector2
        self.flow_orth = Vec2d(self.flow[1], -self.flow[0])
#        if self.flow_orth.dot(self.flow2) < 0:
#            self.flow_orth = -self.flow_orth

        s_idx2 = tri.find_simplex( np.asarray( [current_pos] ) )[0]
        s2 = tri.simplices[s_idx2,:]

        rep_vec = Vec2d()
        min_dist = float("inf")
        min_pt = None
        for i in range(3):
            vtx = array2Vec2d( tri.points[s2[i],:] )
            dist = (current_pos-vtx).get_length()
            if dist < min_dist:
                min_dist = dist
                min_pt = vtx

#        for i in range(3):
#            vtx = array2Vec2d( tri_vor.points[s[i],:] )
#            if s[i] >= num_border_points:
#                # the vertex is part of Voronoi diagram
#                rep_vec += vtx - mean
#                #pass
#            else:
#                # the vertex is part of border
#                #rep_vec -= vtx - mean
#                dist = (current_pos-vtx).get_length()
#                if dist < min_dist:
#                    min_dist = dist

        rep_vec = self.flow_orth
        if (min_pt - current_pos).dot(rep_vec) > 0.0:
            rep_vec = -rep_vec

        threshold_dist = 80.0
        #print min_dist
        if min_dist < threshold_dist:
            force_factor = (threshold_dist-min_dist)/threshold_dist
            rep_vec.normalize_return_length()
            rep_vec = force_factor * rep_vec*2.0
            self.rep_vec = rep_vec
            #print rep_vec
            return rep_vec
        return None


        # TODO; remove the code below:

        dist_map = self.info_spatial_map.dist_map

        range_x = self.info_spatial_map_range.range_x
        steps_x = self.info_spatial_map_range.steps_x
        range_y = self.info_spatial_map_range.range_y
        steps_y = self.info_spatial_map_range.steps_y

        position = self.info_robot_pose.position

        position_img = (int((float(position[0])-range_x[0])*steps_x/(range_x[1] - range_x[0])),
                        int((float(position[1])-range_y[0])*steps_y/(range_y[1] - range_y[0])))

        flow = self.info_flow_vector.flow_vector
        flow_orth = Vec2d(flow[1], -flow[0])
        # get the point with the geatest distance from obstacles along axis orthogonal to flow vector
        if dist_map[position_img[1], position_img[0]] < 60:
            max_dist = dist_map[position_img[1], position_img[0]]
            force_vec = Vec2d()
            dx = int(flow_orth[0]*3)
            dy = int(flow_orth[1]*3)
            ix = position_img[0] + dx
            iy = position_img[1] + dy
            if ix >= 0 and ix < dist_map.shape[1] and\
                    iy >= 0 or iy < dist_map.shape[0]:
                if dist_map[iy,ix] > max_dist:
                    max_dist = dist_map[iy,ix]
                    force_vec = flow_orth
            ix = position_img[0] - dx
            iy = position_img[1] - dy
            if ix >= 0 and ix < dist_map.shape[1] and\
                    iy >= 0 or iy < dist_map.shape[0]:
                if dist_map[iy,ix] > max_dist:
                    max_dist = dist_map[iy,ix]
                    force_vec = -flow_orth
            return force_vec

        return None

    def debugVisDraw(self, debug_info):
        if not self.rep_vec is None and not self.info_robot_pose.position is None and not debug_info is None:
            p1 = self.info_robot_pose.position
            p2 = p1 + self.rep_vec * 50
            debug_info.append( ("vector", "red", p1, p2) )
#            if not self.flow is None:
#                debug_info.append( ("vector", "cyan", p1, p1+self.flow_orth) )
#            if not self.flow2 is None:
#                debug_info.append( ("vector", "magenta", p1, p1+self.flow2*100) )

