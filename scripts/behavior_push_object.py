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

class PushSimulation:
    def __init__(self, mass, moment):
        self.mass = mass
        self.moment = moment

        self.position = Vec2d()
        self.angle = 0.0
        self.velocity = 0.0
        self.angular_velocity = 0.0

    def setState(self, position, angle, velocity, angular_velocity):
        self.position = copy.copy(position)
        self.angle = copy.copy(angle)
        self.velocity = copy.copy(velocity)
        self.angular_velocity = copy.copy(angular_velocity)

    def simulateStep(self, contact_pt_O, contact_force_W, step_size):
        contact_pt_W = contact_pt_O.rotated(self.angle) + self.position
        #contact_force_W = contact_force_O.rotated(self.angle)
        self.velocity += contact_force_W / self.mass * step_size
        torque = (contact_pt_W-self.position).cross(contact_force_W)
        self.angular_velocity += torque / self.moment * step_size

        self.position += self.velocity * step_size
        self.angle += self.angular_velocity * step_size

class BehaviorPushObject:
    def __init__(self):
        self.name = 'push_object'
        self.inputs = ['expected_motion']
        self.outputs = [ 'required_push' ]
        self.inputs_data = {}

        self.best_step = None
        self.best_surf = None
        self.anchored_object_data_name = None
        self.target_point = None

    def update(self, information):
        if not self.anchored_object_data_name is None and not self.anchored_object_data_name in self.inputs:
            self.inputs.append( self.anchored_object_data_name )
        self.inputs_data = extractInformation( information, self.inputs )

        if not (self.inputs[0] in self.inputs_data):# and self.inputs[1] in self.inputs_data):
            return []

        expected_motion = self.inputs_data['expected_motion']
        expected_motion.read = True
        self.anchored_object_data_name = 'perception_' + expected_motion.anchor
        if not self.anchored_object_data_name in self.inputs_data:
            print "BehaviorPushObject: object not found:", expected_motion.anchor, "data name:", self.anchored_object_data_name
            return []

        self.target_point = expected_motion.target_point

        self.obj = self.inputs_data[self.anchored_object_data_name]

        # there are 3 cases:
        # 1) there is a contact and predicted motion is satisfactory, local search is enough
        # 2) there is a contact and predicted motion is not satisfactory, global search is necessary
        # 3) there is no contact

        friction = 0.7
        max_friction_angle = math.atan(friction)
        min_dist = float("inf")
        self.best_step = None
        self.best_surf = None
        self.best_position = None
        self.best_angle = None
        for surf in self.obj.surface:
            pt, n = surf
            n_w = n.rotated(self.obj.angle)
            pt_w = pt.rotated(self.obj.angle) + self.obj.position

            if expected_motion.target_point is None:
                direction_normalized = n_w
            else:
                direction = -(expected_motion.target_point - pt_w)
                abs_angle = abs( (n_w).get_angle_between(direction) )
                if abs_angle > math.pi * 0.4:
                    continue
                friction_needed = math.tan(abs_angle)
                if friction_needed > friction:
                    angle = (n_w).get_angle_between(direction)
                    if angle > max_friction_angle:
                        angle = max_friction_angle
                    elif angle < -max_friction_angle:
                        angle = -max_friction_angle
                    direction = n_w.rotated(angle)
                    #continue
                direction_normalized = direction.normalized()

            sim = PushSimulation(self.obj.mass, self.obj.moment)
            sim.setState(self.obj.position, self.obj.angle, self.obj.velocity, self.obj.angular_velocity)
            for i in range(5):
                #sim.simulateStep(pt, -100.0*n*self.obj.mass, 0.1)
                sim.simulateStep(pt, -10.0*direction_normalized*self.obj.mass, 0.1)
                if expected_motion.target_point is None:
                    dist = abs(wrapAnglePI(expected_motion.target_angle - sim.angle))
                else:
                    dist = (sim.position - expected_motion.target_point).get_length()
                if dist < min_dist:
                    min_dist = dist
                    self.best_step = i
                    self.best_surf = surf
                    self.best_force = direction_normalized
                    self.best_position = copy.copy(sim.position)
                    self.best_angle = copy.copy(sim.angle)

        required_push = InformationRequiredPush()
        required_push.anchor = expected_motion.anchor

        pt, n = self.best_surf
        n_w = self.best_force#n.rotated(self.obj.angle)
        pt_w = pt.rotated(self.obj.angle) + self.obj.position

        required_push.contact_point = pt_w
        required_push.contact_normal = n.rotated(self.obj.angle)
        required_push.contact_force = n_w * 10.0 * self.best_step
        return [ required_push ]

    def debugVisDraw(self, debug_info):
        if not self.best_surf is None and not debug_info is None:
            pt, n = self.best_surf
            pt_w = pt.rotated(self.obj.angle) + self.obj.position
            debug_info.append( ("vector", "yellow", pt_w, pt_w + self.best_force * 50.0) )

            for surf in self.obj.surface:
                pt, n = surf
                p = pt.rotated(self.best_angle) + self.best_position
                debug_info.append( ("circle", "yellow", p, 2) )

            if not self.target_point is None:
                debug_info.append( ("circle", "yellow", self.target_point, 10) )

