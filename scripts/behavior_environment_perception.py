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

class BehaviorEnvironmentPerception:
    def __init__(self, space):
        self.name = 'sim_env_perception'
        self.inputs = []
        self.outputs = [ 'objects_perception' ]
        self.inputs_data = {}

        self.space = space
        self.objects = None

    def update(self, information):

        if self.objects is None:
            self.objects = []
            for body in self.space.bodies:
                for shape in body.shapes:
                    oi = InformationObjectInstance()

                    oi.surface
                    oi.position
                    oi.angle
                    oi.mass
                    oi.moment
                    oi.velocity
                    oi.angular_velocity


        info_inst = InformationObjectInstance()
        info_inst.type = 'instance_' + self.object_name
        info_inst.position = self.object_inst.body.position
        info_inst.angle = self.object_inst.body.angle
        info_inst.velocity = self.object_inst.body.velocity
        info_inst.angular_velocity = self.object_inst.body.angular_velocity
        info_inst.mass = self.object_inst.body.mass
        info_inst.moment = self.object_inst.body.moment
        info_inst.surface = self.object_inst.surface

        return [ info_inst ]

