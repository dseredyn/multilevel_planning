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

# This behavior uses path, spatial_map and current position of robot
# to generate motion that enables the robot to move through tight passages.
class BehaviorTightPassage:
    def __init__(self):
        self.name = 'tight_passage'
        self.inputs = ['robot_pose', 'spatial_map', 'path']
        self.outputs = [ 'robot_control_3' ]

        self.info_robot_pose = None
        self.info_spatial_map = None
        self.info_path = None

        self.path = None

    def update(self, information):
        self.info_robot_pose = None
        self.info_spatial_map = None
        self.info_path = None
        for info in information:
            if info.type == 'robot_pose':
                self.info_robot_pose = info
                info.read = True
            elif info.type == 'spatial_map':
                self.info_spatial_map = info
                info.read = True
            elif info.type == "path":
                self.info_path = info
                info.read = True

        if self.info_robot_pose is None or self.info_spatial_map is None or self.info_path is None:
            self.valid = False
        else:
            self.valid = True

        if not self.valid:
            return []

        if self.info_path.path != self.path:
            # we need to update state of this behavior
            self.path = self.info_path.path

            self.path_clearance = []
            for p_idx in range(len(self.path)-1):
                s_idx = self.path[p_idx]
                n_idx = self.path[p_idx+1]
                pt_idx = self.getCommonEdge(s_idx, n_idx)
                p1 = array2Vec2d(self.info_spatial_map.tri.points[pt_idx[0],:])
                p2 = array2Vec2d(self.info_spatial_map.tri.points[pt_idx[1],:])
                vec = p1-p2
                self.path_clearance.append( (vec.get_length(), p1, p2) )

            radius = 60.0
            self.tight_passages = []
            passage_active = False
            for p_idx in range(len(self.path_clearance)):
                pc = self.path_clearance[p_idx]
                if not passage_active and pc[0] < radius*2.0:
                    passage_active = True
                    passage_start = p_idx
                elif passage_active and pc[0] > radius*2.0:
                    passage_active = False
                    passage_end = p_idx
                    self.tight_passages.append( (passage_start, passage_end) )
            if passage_active:
                passage_end = len(self.path_clearance)-1
                self.tight_passages.append( (passage_start, passage_end) )

            print "tight_passages", self.tight_passages


        current_pos = self.info_robot_pose.position

        current_s_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [current_pos] ) )[0]

        for p_idx in range(len(self.path)-1):
            if current_s_idx == self.path[p_idx]:
                if self.isInTightPassage(p_idx):
                    p1 = self.path_clearance[p_idx][1]
                    p2 = self.path_clearance[p_idx][2]
                    passage = p1-p2

                    robot_dir = Vec2d(1,0).rotated( self.info_robot_pose.angle )
                    angle1 = robot_dir.get_angle_between( passage )
                    angle2 = robot_dir.get_angle_between( -passage )
                    if abs(angle1) < abs(angle2):
                        torque = angle1*4.0
                    else:
                        torque = angle2*4.0
                    #print torque
                    return [ InformationRobotControl3( Vec2d(), torque ) ]

        return []

    def isInTightPassage(self, p_idx):
        for tp in self.tight_passages:
            if p_idx >= tp[0] and p_idx <= tp[1]:
                return True
        return False

    def getCommonEdge(self, s1_idx, s2_idx):
        s1 = self.info_spatial_map.tri.simplices[s1_idx,:]
        s2 = self.info_spatial_map.tri.simplices[s2_idx,:]
        pt_idx = []
        for v1_idx in range(3):
            for v2_idx in range(3):
                if s1[v1_idx] == s2[v2_idx]:
                    pt_idx.append( s1[v1_idx] )
        assert len(pt_idx) == 2
        return pt_idx

