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

# This behavior uses spatial_map, destination and current position of robot
# to generate information about rough path of robot, i.e. list of triangles in
# Delaunay triangulation that are expected to be passed by robot.

#class ClearSpace:
#    def __init__(self):
#        self.clear_points = []
#
#    # every point is given by 2-D coordinate and sigma (for gaussian function)
#    def addClearPoint(self, p):
#        self.clear_points.append( p )
#
#    def getOperations(self, space):
#        query_list = []
#        for cp in self.clear_points:
#            sq = space.segment_query( cp[0], cp[0], cp[1]*2.0, pymunk.ShapeFilter() )
#            query_list += sq
#
#        for q in query_list:
#            if not q.shape is None:
#
#class PushInfo:
#    def __init__(self, n, pos, score, radius):
#        self.n = n
#        self.pos = pos
#        self.score = score
#        self.radius = radius

# 'user-manual' for door
#class ExperienceDoor:
#    def __init__(self):
#        pass
#
#    def getDestPosition(self, clear_space, door_instance_info):
#        # get random angles and choose the best
#        min_penalty = float("inf")
#        best_angle = None
#        for i in range(20):
#            open_angle = random.uniform(0,math.pi/2)
#            angle = open_angle + door_instance_info.angle - door_instance_info.open_angle
#            penalty = 0.0
#            for pt, n, radius in door_instance_info.samples:
#                st = pt.rotated(angle) + door_instance_info.position #(  s[0] * math.cos(angle) + s[1] * math.sin(angle),
#                        #-s[0] * math.sin(angle) + s[1] * math.cos(angle) )
#                for cp, cp_sigma in clear_space.clear_points:
#                    dist = (cp-st).get_length()
#                    #dist = math.sqrt( (cp_x-st[0])**2 + (cp_y-st[1])**2 )
#                    penalty += stats.norm.pdf(dist, 0, cp_sigma)
#            print angle, penalty
#            if penalty < min_penalty:
#                min_penalty = penalty
#                best_angle = open_angle
#
#        #print "best_angle", best_angle
#        return best_angle
#
#    def getPush(self, target_open_angle, door_instance_info, debug_info=None):
#        vec_list = []
#        target_angle = target_open_angle + door_instance_info.angle - door_instance_info.open_angle
#        current_angle = door_instance_info.angle
#        #print target_angle, current_angle
#        for pt, n, radius in door_instance_info.samples:
#            target_st = pt.rotated(target_angle)
#            current_st = pt.rotated(current_angle)
#            if (target_st - current_st).dot(n.rotated(current_angle)) < 0.0:
#                vec_list.append( (current_st, n.rotated(current_angle), radius) )
#        #print "door_instance_info.angle", door_instance_info.angle
#        if debug_info != None:
#            for v in vec_list:
#                debug_info.append( ("vector", "red", v[0]+door_instance_info.position, v[0]+v[1]*50+door_instance_info.position) )
#
#        result = []
#        for pt, n, radius in vec_list:
#            result.append( PushInfo(n, pt+door_instance_info.position, pt.get_length(), radius) )
##            ort = Vec2d(A[1], -A[0])
##            length = ort.get_length()
##            if length > 0:
##                ort = ort / length*100
##            else:
##                continue
##            if ort.dot(B-A) > 0:
##                result.append( (ort, A+door_instance_info.position, A.get_length()) )
##            else:
##                result.append( (-ort, A+door_instance_info.position, A.get_length()) )
#            if debug_info != None:
#                debug_info.append( ("vector", "green", result[-1].pos, result[-1].pos+result[-1].n*30.0) )
#
#        return result
#
#    def predictLeftDoorMovement(self, target_open_angle, door_instance_info, debug_info=None):
#        door_info_copy = copy.copy(door_instance_info)
#        open_angle_diff = target_open_angle - door_instance_info.open_angle
#        steps = int( open_angle_diff / (math.pi/16.0) )
#        steps = max(3, steps)
#        for angle_add in np.linspace(0.0, open_angle_diff, steps, endpoint=False):
#            door_info_copy.angle = door_instance_info.angle + angle_add
#            door_info_copy.open_angle = door_instance_info.open_angle + angle_add
#            push = self.getPush(target_open_angle, door_info_copy)
#            for pu in push:
#                debug_info.append( ("vector", "green", pu[1], pu[1]+pu[0]) )

class DoorInstanceInfo:
    def __init__(self):
        # universal attributes
        self.position = None
        self.angle = None
        self.shape = None
        self.samples = None

        # door-specific attributes
        self.open_angle = None
        self.min_angle = None
        self.max_angle = None
        self.handle_point = None

class ModelStaticWall:
    def __init__(self, a, b, width, friction):
        self.type = "wall"
        self.name = None    # this object cannot be identified by name
        self.a = a
        self.b = b
        self.width = width
        self.friction = friction
        self.static = True
        self.damping = None

        self.surface = []
        step_size = 5.0
        vec = a-b
        n = Vec2d(vec[1], -vec[0])
        n.normalize_return_length()
        steps = max(3, int( vec.get_length()/step_size ))
        for t in np.linspace(0.0, 1.0, steps):
            pt = t*a + (1.0-t)*b + self.width/2.0*n
            self.surface.append( (pt, n) )

        for t in np.linspace(0.0, 1.0, steps):
            pt = t*b + (1.0-t)*a - self.width/2.0*n
            self.surface.append( (pt, n) )

    def addToSpace(self, space):
        self.shape = pymunk.Segment(space.static_body, self.a, self.b, self.width)
        self.shape.friction = self.friction
        self.shape.group = 1
        space.add(self.shape)
        self.body = space.static_body

#    def getPosition(self):
#        return Vec2d()
#
#    def getPosition(self):
#        return Vec2d()

class ModelDoor:
    def __init__(self, name, left, length, width, hinge_pos, rotation, init_angle):
        self.type = "door"
        self.name = name
        self.left = left
        self.length = length
        self.width = width
        self.hinge_pos = hinge_pos
        self.rotation = rotation
        self.init_angle = init_angle
        self.limits = (0.0, math.pi/2.0)
        self.damping = (300, 3000)
        self.static = False

        # polygon for the door
        if self.left:
            self.x_mult = 1.0
        else:
            self.x_mult = -1.0
        p6 = Vec2d(0, -self.width/2.0)
        p7 = Vec2d(0, self.width/2.0)
        p8 = Vec2d(self.x_mult*(self.length/2.0-self.width), self.width/2.0)
        p9 = Vec2d(self.x_mult*(self.length/2.0-self.width), -self.width/2.0)

        self.fp = [p9,p8,p7,p6]

        self.surface = []
        step_size = 5.0
        steps = max(3, int( self.length/2.0/step_size ))
        for t in np.linspace(0.0, 1.0, steps):
            pt = array2Vec2d(t*p7 + (1.0-t)*p8)
            n = Vec2d(0, 1.0)
            self.surface.append( (pt, n) )

        for t in np.linspace(0.0, 1.0, steps):
            pt = array2Vec2d(t*p9 + (1.0-t)*p6)
            n = Vec2d(0, -1.0)
            self.surface.append( (pt, n) )

    def addToSpace(self, space):
        # samples for door
        # each sample contains information about surface point, normal and maximum radius of contacting body
        self.door_samples = []
        for t in np.linspace(0.1, 0.9, 5, endpoint=True):
            self.door_samples.append( (Vec2d(self.x_mult * self.length/2.0 * t, -self.width/2.0), Vec2d(0,-1), float("inf")) )

        radius_tab = [ 100, 50, 15, 5, 5 ]
        idx = 0
        for t in np.linspace(0.1, 0.9, 5, endpoint=True):
            self.door_samples.append( (Vec2d(self.x_mult * self.length/2.0 * t, self.width/2.0), Vec2d(0,1), radius_tab[idx]) )
            idx += 1

        self.door_samples.append( (Vec2d(self.x_mult * (self.length/2.0*0.8 - 4*2.0), 4*4.0), Vec2d(0,1), float("inf")) )
        self.door_samples.append( (Vec2d(self.x_mult * (self.length/2.0*0.8 - 4*2.0), 4*4.0), Vec2d(0,-1), 5) )

        mass = 10.0
        moment = pymunk.moment_for_poly(mass, self.fp)

        self.body = pymunk.Body(mass, moment)
        self.body.position = self.hinge_pos #t5
        self.body.angle = self.rotation + self.init_angle
        self.door_shape = pymunk.Poly(self.body, self.fp, radius=1)
        self.door_shape.group = 1

        # handle
        self.handle_shape1 = pymunk.Segment(self.body, Vec2d(self.x_mult * (self.length/2.0*0.8), self.width*0.5), Vec2d(self.x_mult * (self.length/2.0*0.8), self.width*4.0), self.width)
        self.handle_shape1.group = 1

        self.handle_shape2 = pymunk.Segment(self.body, Vec2d(self.x_mult * (self.length/2.0*0.8 - self.width*4.0), self.width*4.0), Vec2d(self.x_mult * (self.length/2.0*0.8), self.width*4.0), self.width)
        self.handle_shape2.group = 1

        space.add(self.body, self.door_shape, self.handle_shape1, self.handle_shape2)

        j = pymunk.PivotJoint(self.body, space.static_body, self.body.position)
        s = pymunk.DampedRotarySpring(self.body, space.static_body, 0, 0, 90000)
        space.add(j, s)

        j2 = pymunk.RotaryLimitJoint(self.body, space.static_body, -(self.rotation+self.limits[1]), -(self.rotation+self.limits[0]))
        space.add(j2)

    def getOpenAngle(self):
        return self.body.angle - self.rotation

class ModelCabinetCase:
    def __init__(self, name, width, depth, position, rotation):
        self.type = "cabinet_case"
        self.name = name
        self.width = width
        self.depth = depth
        self.position = position
        self.rotation = rotation
        self.damping = None
        self.line_width = 4.0

        p5 = Vec2d(-self.width/2.0, self.depth/2.0 + self.line_width*2)     # left door joint
        self.hinge_pos_W = p5.rotated(self.rotation) + self.position #transform(p5, self.position, self.rotation)

        p1 = Vec2d(-self.width/2.0, self.depth/2.0)
        p2 = Vec2d(-self.width/2.0, -self.depth/2.0)
        p3 = Vec2d(self.width/2.0, -self.depth/2.0)
        p4 = Vec2d(self.width/2.0, self.depth/2.0)

        self.t1 = p1.rotated(self.rotation) + self.position
        self.t2 = p2.rotated(self.rotation) + self.position
        self.t3 = p3.rotated(self.rotation) + self.position
        self.t4 = p4.rotated(self.rotation) + self.position

        friction = 0.5
        self.left_door = ModelDoor(self.name + "_left_door", True, self.width, self.line_width, self.hinge_pos_W, self.rotation, 0)
        self.models = [
            ModelStaticWall(self.t1, self.t2, self.line_width, friction),
            ModelStaticWall(self.t2, self.t3, self.line_width, friction),
            ModelStaticWall(self.t3, self.t4, self.line_width, friction),
            self.left_door,
        ]

    def getModels(self):
        return self.models

    def addToSpace(self, space):
        # this is meta model
        pass
#        static= [pymunk.Segment(space.static_body, t1, t2, self.line_width)
#                    ,pymunk.Segment(space.static_body, t2, t3, self.line_width)
#                    ,pymunk.Segment(space.static_body, t3, t4, self.line_width)
#                    ]  
#        for s in static:
#            s.friction = 1.
#            s.group = 1
#        space.add(static)
        #self.left_door.addToSpace( space )

    def debugVis(self, debug_info=None):
        if not debug_info is None:
            for p, n, r in self.door_samples:
                p1 = p.rotated(self.l_door_body.angle) + self.l_door_body.position
                p2 = (p+n*30.0).rotated(self.l_door_body.angle) + self.l_door_body.position
                debug_info.append( ("vector", "green", p1, p2) )
                if r != float("inf"):
                    p3 = (p + n/n.get_length()*r).rotated(self.l_door_body.angle) + self.l_door_body.position
                    debug_info.append( ("circle", "red", p3, r) )

    def getLeftDoorInfo(self):
        result = DoorInstanceInfo()
        # universal attributes
        result.position = self.l_door_body.position
        result.angle = self.l_door_body.angle
        result.shape = self.fp

        result.samples = self.door_samples

        # door-specific attributes
        result.open_angle = result.open_angle = self.l_door_body.angle - self.rotation
        result.min_angle = 0.0
        result.max_angle = math.pi/2.0
        #result.handle_point = Vec2d(self.width/2.0*0.8 - 4*2.0, 4*4.0)

        return result

class ModelBox:
    def __init__(self, name, width_x, width_y, position, angle, damping=None):
        self.type = "object"
        self.name = name
        self.mass = 10.0
        self.width_x = width_x
        self.width_y = width_y
        self.position = position
        self.angle = angle
        self.static = False
        self.damping = damping

        self.fp = [
            Vec2d(-0.5*width_x, -0.5*width_y),
            Vec2d(0.5*width_x, -0.5*width_y),
            Vec2d(0.5*width_x, 0.5*width_y),
            Vec2d(-0.5*width_x, 0.5*width_y),
        ]
        step_size = 5.0
        self.surface = []
        t_min = 0.0
        t_max = 1.0
        t_steps_x = max(3, int(width_x/step_size))
        t_steps_y = max(3, int(width_y/step_size))
        for t in np.linspace(t_min, t_max, t_steps_x, endpoint=True):
            pt = Vec2d(-0.5*width_x + t*width_x, -0.5*width_y)
            n = Vec2d(0,-1)
            self.surface.append( (pt, n) )

        for t in np.linspace(t_min, t_max, t_steps_y, endpoint=True):
            pt = Vec2d(0.5*width_x, -0.5*width_y + t*width_y)
            n = Vec2d(1,0)
            self.surface.append( (pt, n) )

        for t in np.linspace(t_min, t_max, t_steps_x, endpoint=True):
            pt = Vec2d(0.5*width_x - t*width_x, 0.5*width_y)
            n = Vec2d(0,1)
            self.surface.append( (pt, n) )

        for t in np.linspace(t_min, t_max, t_steps_y, endpoint=True):
            pt = Vec2d(-0.5*width_x, 0.5*width_y - t*width_y)
            n = Vec2d(-1,0)
            self.surface.append( (pt, n) )

    def addToSpace(self, space):
        moment = pymunk.moment_for_poly(self.mass, self.fp)
        self.body = pymunk.Body(self.mass, moment)
        self.body.position = self.position
        self.body.angle = self.angle
        self.shape = pymunk.Poly(self.body, self.fp, radius=1)
        self.shape.group = 1

        self.shape.friction = 1.

        space.add(self.body, self.shape)

class ModelRobot:
    def __init__(self, position, angle):
        self.type = "robot"
        self.name = "robot"
        self.mass = 10
        self.radius = 20
        self.inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0,0))
        self.position = position
        self.angle = angle
        self.damping = None

        self.arm_l_0 = Vec2d(0, -40)
        self.arm_l_1 = Vec2d(15, -40)
        self.arm_r_0 = Vec2d(0, 40)
        self.arm_r_1 = Vec2d(15, 40)
        self.arm_width = 3

        self.qhull = [  Vec2d(self.radius, 0.0),
                        self.arm_l_1 + Vec2d(self.arm_width, -self.arm_width),
                        self.arm_l_0 + Vec2d(-self.arm_width, -self.arm_width),
                        Vec2d(-self.radius, 0.0).rotated(0.3),
                        Vec2d(-self.radius, 0.0),
                        Vec2d(-self.radius, 0.0).rotated(-0.3),
                        self.arm_r_0 + Vec2d(-self.arm_width, self.arm_width),
                        self.arm_r_1 + Vec2d(self.arm_width, self.arm_width), ]

        self.outer_radius = self.qhull[0].get_length()
        for p in self.qhull[1:]:
            if p.get_length() > self.outer_radius:
                self.outer_radius = p.get_length()

        # pushing capabilities
        self.push_capabilities = []
        # body
        idx = 0
        radius_tab = [  float("inf"), 40.0, 7.0, 1.0, 50.0, float("inf"),
                        float("inf"), 50.0, 1.0, 7.0, 40.0, float("inf"),]
        for alpha in np.linspace(0.0, 2.0*math.pi, 12):
            contact = Vec2d(self.radius, 0.0).rotated(alpha)
            force = contact / contact.get_length()
            self.push_capabilities.append( (contact, force, radius_tab[idx] ) )
            idx += 1

        # arms
        # left arm
        # outwards
        contact = self.arm_l_0
        force = contact / contact.get_length()
        self.push_capabilities.append( (contact, force, float("inf") ) )

        contact = self.arm_l_1
        force = contact / contact.get_length()
        self.push_capabilities.append( (contact, force, float("inf") ) )

        # inwards
        contact = self.arm_l_0*0.3 + self.arm_l_1*0.7
        force = -contact / contact.get_length()
        self.push_capabilities.append( (contact, force, 7.0 ) )     # the last attribute is maximum radius of another object

        # right arm
        # outwards
        contact = self.arm_r_0
        force = contact / contact.get_length()
        self.push_capabilities.append( (contact, force, float("inf") ) )

        contact = self.arm_r_1
        force = contact / contact.get_length()
        self.push_capabilities.append( (contact, force, float("inf") ) )

        # inwards
        contact = self.arm_r_0*0.3 + self.arm_r_1*0.7
        force = -contact / contact.get_length()
        self.push_capabilities.append( (contact, force, 7.0 ) )

    def addToSpace(self, space):
        self.robot_body = pymunk.Body(self.mass, self.inertia)
        self.robot_shape = pymunk.Circle(self.robot_body, self.radius)
        self.robot_shape.color = (255,50,50)
        self.robot_shape.filter = pymunk.ShapeFilter(group=2)
        self.shoulders = pymunk.Segment(self.robot_body, self.arm_l_0, self.arm_r_0, self.arm_width)
        self.shoulders.filter = pymunk.ShapeFilter(group=2)
        self.arm_l = pymunk.Segment(self.robot_body, self.arm_l_0, self.arm_l_1, self.arm_width)
        self.arm_l.filter = pymunk.ShapeFilter(group=2)
        self.arm_r = pymunk.Segment(self.robot_body, self.arm_r_0, self.arm_r_1, self.arm_width)
        self.arm_r.filter = pymunk.ShapeFilter(group=2)

        self.shoulders.friction = 1.
        self.arm_l.friction = 1.
        self.arm_r.friction = 1.
        self.robot_shape.friction = 1.

        self.robot_body.position = self.position
        self.robot_body.angle = self.angle
        space.add(self.robot_body, self.robot_shape, self.shoulders, self.arm_l, self.arm_r)

    def debugVisQhull(self, debug_info=None):
        if not debug_info is None:
            for i in range(len(self.qhull)):
                p1 = self.qhull[i].rotated(self.robot_body.angle) + self.robot_body.position
                p2 = self.qhull[(i+1)%len(self.qhull)].rotated(self.robot_body.angle) + self.robot_body.position
                debug_info.append( ("line", "yellow", p1, p2) )

    def debugVisPushing(self, debug_info=None):
        if not debug_info is None:
            for contact, force, radius in self.push_capabilities:
                p1 = contact.rotated(self.robot_body.angle) + self.robot_body.position
                p2 = (contact + 20.0*force).rotated(self.robot_body.angle) + self.robot_body.position
                debug_info.append( ("vector", "green", p1, p2) )
                #if radius != float("inf"):
                #    p3 = (contact + force/force.get_length()*radius).rotated(self.robot_body.angle) + self.robot_body.position
                #    debug_info.append( ("circle", "green", p3, radius) )

    def debugVisPose(self, debug_info, pos, angle):
        if not debug_info is None:
            debug_info.append( ("line", "red", self.arm_l_0.rotated(angle) + pos, self.arm_l_1.rotated(angle) + pos) )
            debug_info.append( ("line", "red", self.arm_l_0.rotated(angle) + pos, self.arm_r_0.rotated(angle) + pos) )
            debug_info.append( ("line", "red", self.arm_r_1.rotated(angle) + pos, self.arm_r_0.rotated(angle) + pos) )
            debug_info.append( ("circle", "red", pos, self.radius) )

#class TaskTightPassages:
#    def __init__(self, robot, path, space_map):
#        self.robot = robot
#        tight_passage = False
#        min_clearance = float("inf")
#        min_edge = None
#        passages = []
#        for i in range(len(path)-1):
#            pt_idx = space_map.getCommonEdge(path[i], path[i+1])
#            A = array2Vec2d( space_map.tri.points[pt_idx[0],:])
#            B = array2Vec2d( space_map.tri.points[pt_idx[1],:])
#            clearance = (A-B).get_length()
#            #ori = self.getOrientationPossibilities(A, B, clearance)
#            #print path[i], path[i+1], clearance
#            if clearance < robot.outer_radius*3.0:
#                if not tight_passage:
#                    tight_passage = True
#                    tight_begin = i
#                else:
#                    tight_end = i
#                if 
#                #TODO
#                min_clearance = min(min_clearance, clearance)
#                min_edge = (A, B)
#            else:
#                if tight_passage:
#                    tight_passage = False
#                    passages.append( (tight_begin, tight_end, min_clearance) )
#                    min_clearance = float("inf")
#        print "passages", passages
#
#    def getOrientationPossibilities(self, clearance, A, B):
#        # for the robot there are usually two possible orientations with some error margin and
#        # possible transition between them with given likehood
#        result = ProbabilityGraph()
#        if clearance > self.robot.outer_radius*2.5:
#            result.v[0] = ("uniform", 0.0, 2.0*math.pi)
#        else:
#            result.v[0] = ("normal", wrapAnglePI((A-B).get_angle()+math.pi/2.0), math.pi/16.0)
#            result.v[1] = ("normal", wrapAnglePI((A-B).get_angle()-math.pi/2.0), math.pi/16.0)
#            prob = (clearance - self.robot.outer_radius*2.0) / (self.robot.outer_radius*2.5 - self.robot.outer_radius*2.0)
#            prob = min(1.0, max(0.0, prob))
#            result.addEdge(0, 1, prob)
#            result.addEdge(1, 0, prob)
#        return result

