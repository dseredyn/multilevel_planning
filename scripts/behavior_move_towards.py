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
#
# Opis zachowania:
# Rozpatrzmy 2 trojkaty z triangulacji Delone: dla biezacej lokalizacji robota i dla nastepnej w sciezce.
# Zachowanie generuje sile w kierunku prostopadlym do wspolnej krawedzi dla obu trojkatow. Sila w kierunku
# ortogonalnym jest dowolna, podobnie jak moment w osi rotacji.
#
class BehaviorMoveTowards:
    def __init__(self):
        # common variables for all behaviors
        self.name = 'move_towards'
#        self.inputs = [ 'robot_pose', 'destination_geom', 'spatial_map', 'path' ]
        self.inputs = [ 'robot_pose', 'destination_geom', 'path' ]
        self.outputs = [ 'robot_control_1', 'flow_vector', 'stuck' ]

        # input variables
        self.info_robot_pose = None
        self.info_destination_geom = None
#        self.info_spatial_map = None
        self.info_path = None

        # state variables
        self.robot_pose_history = None
        self.robot_pose_history_idx = 0
        self.max_robot_pose_history_length = 100
        self.path = None
        self.min_angle = None
        self.max_angle = None

    def update(self, information):
        self.info_robot_pose = None
        self.info_destination_geom = None
#        self.info_spatial_map = None
        self.info_path = None
        for info in information:
            if info.type == 'robot_pose':
                self.info_robot_pose = info
                info.read = True
            elif info.type == 'destination_geom':
                self.info_destination_geom = info
                info.read = True
#            elif info.type == 'spatial_map':
#                self.info_spatial_map = info
#                info.read = True
            elif info.type == 'path':
                self.info_path = copy.deepcopy(info)
                info.read = True

#        if self.info_robot_pose is None or self.info_destination_geom is None or self.info_spatial_map is None or self.info_path is None:
        if self.info_robot_pose is None or self.info_destination_geom is None or self.info_path is None:
            self.valid = False
        else:
            self.valid = True

        if not self.valid:
            return []

        if self.info_path.path != self.path:
            # the path has changed, so we need to update some state variables
            self.robot_pose_history = []
            self.robot_pose_history_idx = 0
            self.path = self.info_path.path


        if len(self.robot_pose_history) < self.max_robot_pose_history_length:
            self.robot_pose_history.append( (self.info_robot_pose.position, self.info_robot_pose.angle) )
            self.robot_pose_history_idx = len(self.robot_pose_history) - 1
        else:
            self.robot_pose_history_idx = (self.robot_pose_history_idx+1)%self.max_robot_pose_history_length
            self.robot_pose_history[self.robot_pose_history_idx] = (self.info_robot_pose.position, self.info_robot_pose.angle)

        #print len(self.robot_pose_history)
        # check if the robot has stuck
        if len(self.robot_pose_history) == self.max_robot_pose_history_length:
            old_idx = (self.robot_pose_history_idx+1)%self.max_robot_pose_history_length
            pose_old = self.robot_pose_history[old_idx]
            #print pose_old[0]-self.info_robot_pose.position
            if (pose_old[0]-self.info_robot_pose.position).get_length() < 1.0:
                #print "stuck"
                return [ InformationRobotHasStuck() ]

        force, torque, flow_vector2 = self.getDrivingForce()

        if force is None:
            return []

        torque = torque*0.5
        self.force = force
        self.torque = torque

        info_flow = InformationFlow(force)
#        info_flow.flow_vector2 = flow_vector2

        return [ InformationRobotControl1(force, torque), info_flow ]

    def getDrivingForce(self):
        current_pos = self.info_robot_pose.position
        current_angle = self.info_robot_pose.angle
        target_pos = self.info_destination_geom.point
        target_robot_angle = self.info_destination_geom.angle
        path = self.info_path.path

        #source_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [current_pos] ) )[0]

        #print self.info_spatial_map.tri.find_simplex( np.asarray( [current_pos] ) )
        #s_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [current_pos] ), bruteforce=True )[0]
#        s_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [current_pos] ) )[0]
        s_idx = self.info_path.tri.find_simplex( np.asarray( [current_pos] ) )[0]

        path_pos = None
        for p_idx in range(len(path)):
            if path[p_idx] == s_idx:
                path_pos = p_idx
                break

        if target_robot_angle is None:
            angle_diff = 0.0
        else:
            angle_diff = wrapAnglePI( target_robot_angle - current_angle )

        if path_pos is None:
            #raise Exception("robot is outside path")
            n_idx = self.info_path.prev[s_idx]
            #print s_idx, n_idx
            pt_idx = self.getCommonEdge(s_idx, n_idx)
            if pt_idx is None:
                return None, None, None
            a = array2Vec2d(self.info_path.tri.points[pt_idx[0],:])
            b = array2Vec2d(self.info_path.tri.points[pt_idx[1],:])
            n = a - b
            n = 0.1*Vec2d(n[1], -n[0])
            if n.get_length() > 1:
                n.normalize_return_length()
#            n2 = self.info_spatial_map.meancenters[s_idx] - self.info_spatial_map.meancenters[n_idx]
            n2 = self.info_path.meancenters[s_idx] - self.info_path.meancenters[n_idx]
            if n.dot(n2) > 0:
                n = -n
            return n, angle_diff, n

        if path_pos == len(path)-1:
            n = target_pos - current_pos
            n *= 0.1
            if n.get_length() > 1.0:
                n = n / n.get_length()
            return n, angle_diff, n

        flow_vector2 = None
        pt_idx = self.getCommonEdge(path[path_pos], path[path_pos+1])
        if pt_idx is None:
            return None, None, None
        a = array2Vec2d(self.info_path.tri.points[pt_idx[0],:]) - current_pos
        b = array2Vec2d(self.info_path.tri.points[pt_idx[1],:]) - current_pos
        #vec = (a-b)
        #flow_vector2 = Vec2d(vec[1], -vec[0])
        #flow_vector2 = 0.5*(a+b)
        flow_vector2 = array2Vec2d(self.info_path.circumcenters[path[path_pos+1],:]) - array2Vec2d(self.info_path.circumcenters[path[path_pos],:])
#        n2 = self.info_path.meancenters[path[path_pos]] - self.info_path.meancenters[path[path_pos+1]]
#        if flow_vector2.dot(n2) > 0:
#            flow_vector2 = -flow_vector2

        radius = 45.0
        min_angle = None
        max_angle = None
        for p_idx in range(path_pos, len(path)-1):
            pt_idx = self.getCommonEdge(path[p_idx], path[p_idx+1])
            if pt_idx is None:
                return None, None, None
            a = array2Vec2d(self.info_path.tri.points[pt_idx[0],:]) - current_pos
            b = array2Vec2d(self.info_path.tri.points[pt_idx[1],:]) - current_pos
            # express points in polar coordinates
            angle1 = wrapAngle2PI( a.get_angle() - b.get_angle() )
            angle2 = wrapAngle2PI( b.get_angle() - a.get_angle() )
            stop = False
            if angle1 < angle2:
                if radius/b.get_length()+radius/a.get_length() > abs(b.get_angle_between(a)):
                    a_angle = b.get_angle() + b.get_angle_between(a)*0.5
                    b_angle = a_angle
                else:
                    b_angle = b.get_angle() + radius/b.get_length()
                    a_angle = a.get_angle() - radius/a.get_length()
                if min_angle is None:
                    min_angle = b_angle
                    max_angle = a_angle
                else:
                    min_angle_diff = wrapAnglePI( b_angle - min_angle )
                    if min_angle_diff > 0.0:
                        min_angle = b_angle
                        if wrapAnglePI(max_angle - min_angle) < 0:
                            min_angle = max_angle
                            stop = True
                    max_angle_diff = wrapAnglePI( a_angle - max_angle )
                    if max_angle_diff < 0.0:
                        max_angle = a_angle
                        if wrapAnglePI(max_angle - min_angle) < 0:
                            max_angle = min_angle
                            stop = True
            else:
                if radius/b.get_length()+radius/a.get_length() > abs(b.get_angle_between(a)):
                    a_angle = b.get_angle() + b.get_angle_between(a)*0.5
                    b_angle = a_angle
                else:
                    b_angle = b.get_angle() - radius/b.get_length()
                    a_angle = a.get_angle() + radius/a.get_length()
                if min_angle is None:
                    min_angle = a_angle
                    max_angle = b_angle
                else:
                    min_angle_diff = wrapAnglePI( a_angle - min_angle )
                    if min_angle_diff > 0.0:
                        min_angle = a_angle
                        if wrapAnglePI(max_angle - min_angle) < 0:
                            min_angle = max_angle
                            stop = True
                    max_angle_diff = wrapAnglePI( b_angle - max_angle )
                    if max_angle_diff < 0.0:
                        max_angle = b_angle
                        if wrapAnglePI(max_angle - min_angle) < 0:
                            max_angle = min_angle
                            stop = True
            min_angle = wrapAnglePI(min_angle)
            max_angle = wrapAnglePI(max_angle)
            if stop:
                break
        target_angle = (target_pos - current_pos).get_angle()

        min_angle_diff = wrapAnglePI( target_angle - min_angle )
        if min_angle_diff > 0.0:
            min_angle = target_angle
            if wrapAnglePI(max_angle - min_angle) < 0:
                min_angle = max_angle
        max_angle_diff = wrapAnglePI( target_angle - max_angle )
        if max_angle_diff < 0.0:
            max_angle = target_angle
            if wrapAnglePI(max_angle - min_angle) < 0:
                max_angle = min_angle

        self.min_angle = min_angle
        self.max_angle = max_angle

        flow_vector = Vec2d(1,0).rotated(min_angle)

        return flow_vector, angle_diff, flow_vector2

    def debugVisDraw(self, debug_info):
        if not self.force is None and not debug_info is None:
            current_pos = self.info_robot_pose.position
            debug_info.append( ("vector", "blue", current_pos, current_pos + self.force * 50.0) )
            if not self.min_angle is None:
                debug_info.append( ("line", "green", current_pos, current_pos + Vec2d(1,0).rotated(self.min_angle) * 80.0) )
            if not self.max_angle is None:
                debug_info.append( ("line", "red", current_pos, current_pos + Vec2d(1,0).rotated(self.max_angle) * 80.0) )

    def getDrivingForce_locally(self):
        raise Exception('this method should not be used')
        current_pos = self.info_robot_pose.position
        target_pos = self.info_destination_geom.point
        path = self.info_path.path

        #source_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [current_pos] ) )[0]

        #print self.info_spatial_map.tri.find_simplex( np.asarray( [current_pos] ) )
        s_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [current_pos] ), bruteforce=True )[0]
        n_idx = -1
        last_index = False
        for p_idx in range(len(path)):
            if path[p_idx] == s_idx:
                if p_idx + 1 < len(path):
                    n_idx = path[p_idx + 1]
                else:
                    last_index = True
                break
        if last_index:
            n = target_pos - current_pos
            n *= 0.1
            if n.get_length() > 1.0:
                n = n / n.get_length()
            return n
        elif n_idx == -1:
            print "outside path"
            return None
        else:
            pt_idx = self.getCommonEdge(s_idx, n_idx)
            p1 = array2Vec2d( self.info_spatial_map.tri.points[pt_idx[0],:] )
            p2 = array2Vec2d( self.info_spatial_map.tri.points[pt_idx[1],:] )
            vec = p1-p2

            n = Vec2d( vec[1], -vec[0] )
            n = n / n.get_length()
            s1_s2_vec = self.info_spatial_map.meancenters[n_idx] - self.info_spatial_map.meancenters[s_idx]
            if n.dot(s1_s2_vec) > 0:
                return n
            else:
                return -n
        return None

    def getCommonEdge(self, s1_idx, s2_idx):
#        s1 = self.info_spatial_map.tri.simplices[s1_idx,:]
#        s2 = self.info_spatial_map.tri.simplices[s2_idx,:]
        s1 = self.info_path.tri.simplices[s1_idx,:]
        s2 = self.info_path.tri.simplices[s2_idx,:]
        pt_idx = []
        for v1_idx in range(3):
            for v2_idx in range(3):
                if s1[v1_idx] == s2[v2_idx]:
                    pt_idx.append( s1[v1_idx] )
        #assert len(pt_idx) == 2
        if len(pt_idx) == 2:
            return pt_idx
        else:
            print "pt_idx", pt_idx, s1_idx, s2_idx
            return None




def shortcutAlgorithmTest():
    points = [
        (3, 1),
        (1, 2),
        (4, 3),
        (1, 5),
        (11,1),
        (5, 6),
        (10,5),
        (8, 8),
        (10,9),
        (4,9),
        (6,12),
    ]

    radius = 10.5

    for idx in range(len(points)):
        points[idx] = Vec2d(points[idx][0], points[idx][1]).rotated(3*math.pi/4.0)

    min_angle = None
    max_angle = None
    for p_idx in range(len(points)-1):
        a = Vec2d(points[p_idx][0], points[p_idx][1])
        b = Vec2d(points[p_idx+1][0], points[p_idx+1][1])
        # express points in polar coordinates
        angle1 = wrapAngle2PI( a.get_angle() - b.get_angle() )
        angle2 = wrapAngle2PI( b.get_angle() - a.get_angle() )
        print angle1, angle2
        stop = False
        if angle1 < angle2:
            if radius/b.get_length()+radius/a.get_length() > abs(b.get_angle_between(a)):
                a_angle = b.get_angle() + b.get_angle_between(a)*0.5
                b_angle = a_angle
            else:
                b_angle = b.get_angle() + radius/b.get_length()
                a_angle = a.get_angle() - radius/a.get_length()
            if min_angle is None:
                min_angle = b_angle
                max_angle = a_angle
            else:
                min_angle_diff = wrapAnglePI( b_angle - min_angle )
                if min_angle_diff > 0.0:
                    min_angle = b_angle
                    if wrapAnglePI(max_angle - min_angle) < 0:
                        min_angle = max_angle
                        stop = True
                max_angle_diff = wrapAnglePI( a_angle - max_angle )
                if max_angle_diff < 0.0:
                    max_angle = a_angle
                    if wrapAnglePI(max_angle - min_angle) < 0:
                        max_angle = min_angle
                        stop = True
        else:
            if radius/b.get_length()+radius/a.get_length() > abs(b.get_angle_between(a)):
                a_angle = b.get_angle() + b.get_angle_between(a)*0.5
                b_angle = a_angle
            else:
                b_angle = b.get_angle() - radius/b.get_length()
                a_angle = a.get_angle() + radius/a.get_length()
            if min_angle is None:
                min_angle = a_angle
                max_angle = b_angle
            else:
                min_angle_diff = wrapAnglePI( a_angle - min_angle )
                if min_angle_diff > 0.0:
                    min_angle = a_angle
                    if wrapAnglePI(max_angle - min_angle) < 0:
                        min_angle = max_angle
                        stop = True
                max_angle_diff = wrapAnglePI( b_angle - max_angle )
                if max_angle_diff < 0.0:
                    max_angle = b_angle
                    if wrapAnglePI(max_angle - min_angle) < 0:
                        max_angle = min_angle
                        stop = True
        min_angle = wrapAnglePI(min_angle)
        max_angle = wrapAnglePI(max_angle)
#            if wrapAnglePI(max_angle - min_angle) < 0.1:
#                break

        plt.plot( [p[0] for p in points], [p[1] for p in points], "r" )
        plt.plot( [a[0],b[0]], [a[1],b[1]], "b" )
        plt.plot( [0,math.cos(min_angle)*10], [0,math.sin(min_angle)*10], "g" )
        plt.plot( [0,math.cos(max_angle)*10], [0,math.sin(max_angle)*10], "y" )
        plt.show()
        if stop:
            break

