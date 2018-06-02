#!/usr/bin/env python

import sys
import math
import numpy as np
import random
import time

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

from information import *
from behavior_spatial_map import *
from behavior_voronoi import *
from behavior_move_towards import *
from behavior_obstacle_avoidance import *
from behavior_robot_control import *
from behavior_tight_passage import *
from behavior_open_door import *
from behavior_move_object import *
from behavior_object_perception import *
from behavior_push_object import *
from behavior_keep_contact import *
from behavior_push_execution import *
from behavior_space_digging import *
from manual_control import *
from simulation_models import *
from environment_xml import *

def generateBehaviorsDotGraph(filename, behaviors):
    with open(filename, "w") as f:
        graph_str = "digraph behaviors {\n"
        data_type_names = set()
        for b in behaviors:
            graph_str += "    behavior_" + b.name + ' [label="' + b.name + '"];\n'
            for inp in b.inputs:
                data_type_names.add( inp )
                graph_str += "    " + inp + " -> behavior_" + b.name + ";\n"

            for outp in b.outputs:
                data_type_names.add( outp )
                graph_str += "    behavior_" + b.name + " -> " + outp + ";\n"

        for data_type in data_type_names:
            graph_str += "    " + data_type + " [shape=rectangle];\n"
        graph_str += "}\n"
        f.write( graph_str )

def main():
    sys.setrecursionlimit(10000)

    width, height = 690,600

    ### PyGame init
    pygame.init()
    screen = pygame.display.set_mode((width,height)) 
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont("Arial", 16)
    
    ### Physics stuff
    space = pymunk.Space()   
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    env_xml = EnvironmentXml("/home/dseredyn/ws_stero/src/multilevel_planning/data/environments/env_push.xml")
    models = env_xml.getModels()

    for model in models:
        model.addToSpace(space)

    damped_objects_list = []
    for model in models:
        if not model.damping is None:
            damped_objects_list.append( (model.body, model.damping[0], model.damping[1]) )

#    cab = ModelCabinet(200, 100, Vec2d(200,200), -0.05)
#    cab.addToSpace(space)

#    box = ModelBox(50, 10, Vec2d(200,200), 0*math.pi/4.0)
#    box.addToSpace( space )

#    damped_objects_list.append( (box.body, 300.0, 3000.0) )

    # robot
#    rob = ModelRobot( Vec2d(100,100), 0.0 )
#    rob = ModelRobot( Vec2d(220,225), 0.0 )
#    rob.addToSpace(space)
    for ob in models:
        if ob.name == "robot":
            rob = ob
        if ob.name == "box":
            box = ob

    control_mode = "target_pos"
    position_control = ManualPositionControl(1000.0, 500.0, 50000.0, 50000.0)

    control_mode = "auto"

    dest_point = Vec2d(550,270)

#    info_exp_motion = InformationExpectedMotion()
#    info_exp_motion.anchor = "instance_box"
#    info_exp_motion.target_point = Vec2d(500,500)

    move_command = InformationMoveObjectCommand("box", Vec2d(500,500))
    # provide information
    information = [
        #InformationSpatialMapRange( (0, width), 100, (0, height), 100 ),        # range and resolution of spatial map
        InformationRobotPose( rob.robot_body.position, rob.robot_body.angle ),  # current pose of robot
        InformationPerceptionSpace( space, models ),                                    # percepted environment (exact, full perception)
#        InformationDestinationGeom( dest_point ),                               # target point for motion
#        InformationOpenDoorCommand(),
#        info_exp_motion,
        move_command,
    ]

    behaviors = [
        BehaviorSpatialMapGeneration(models),
#        BehaviorSpaceDigging(env),
        BehaviorKeepContact(),
        BehaviorSpatialPathPlanner(),
        BehaviorMoveTowards(),
        BehaviorObstacleAvoidance(),
        BehaviorTightPassage(),
#        BehaviorDoorPerception(cab.left_door),
#        BehaviorOpenDoor(),
        BehaviorMoveObject(),
        BehaviorObjectPerception(),#box, "box"),
        BehaviorPushObject(),
        BehaviorPushExecution(),
        BehaviorRobotControl(),
    ]

#    behaviors[0].update(information)
#    behaviors[0].plotTriangulation()
#    behaviors[0].plotGraph()
#    plt.show()
#    behaviors[1].update(information)
#    behaviors[1].plotTriangulation()
#    behaviors[1].plotGraph()
#    behaviors[1].plotOccludedSimplices()
#    behaviors[1].plotBorder()
#    plt.show()
#    exit(0)

    generateBehaviorsDotGraph('/home/dseredyn/svn/phd/ds/doktorat/rozwazania_2018_04/img/zachowania.dot', behaviors)

    first_behavior_iteration = True
    iterations = 0
    pause = False
    show_debug = False
    while running:
        iterations += 1
        if iterations == 10:
            generateBehaviorsDotGraph('/home/dseredyn/svn/phd/ds/doktorat/rozwazania_2018_04/img/zachowania_2.dot', behaviors)

        for event in pygame.event.get():
            if event.type == QUIT or \
                event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                running = False
            if event.type == KEYDOWN and event.key == K_SPACE:
                pause = not pause
            if event.type == KEYDOWN and event.key == K_d:
                show_debug = not show_debug

        if pause:
            time.sleep(0.1)
            continue

        debug_info = []

        keys = pygame.key.get_pressed()

        active_behaviors = []

        # manual control
        if control_mode == "force":
            manualForceControl(rob.robot_body, keys)
        elif control_mode == "target_pos":
            position_control.update( rob.robot_body, keys )
        elif control_mode == "auto":
            position_control.update( rob.robot_body, keys )
            joinInformation( information, [ InformationRobotPose( rob.robot_body.position, rob.robot_body.angle ) ] )   # current pose of robot

            for b in behaviors:
                new_inf = b.update( information )
                joinInformation( information, new_inf )
                if len(new_inf) > 0:
                    active_behaviors.append( b.name )
                    #print new_inf[0].type
            inf_list = []
            for inf in information:
                inf_list.append( inf.type )
                if inf.type == "robot_total_control":
                    lin_damping = 300.0
                    lin_stiffness = 5000.0
                    rot_damping = 5000.0
                    rot_stiffness = 5000.0
                    #print inf.force, inf.torque
                    rob.robot_body.force = lin_stiffness*inf.force - lin_damping * rob.robot_body.velocity
                    rob.robot_body.torque = rot_stiffness*inf.torque - rot_damping * rob.robot_body.angular_velocity
            #print inf.torque
            #print "data:", inf_list
            # TODO: fix removing old information wrt. behavior execution sequence
            clearObsoleteInformation( information )
            #print active_behaviors
#        print "plan_idx", plan_idx
        #cab.debugVis(debug_info)

        #rob.debugVisQhull(debug_info)
        #rob.debugVisPushing(debug_info)

        applyDamping( damped_objects_list )

        mouse_position = pymunk.pygame_util.from_pygame( Vec2d(pygame.mouse.get_pos()), screen )
        mouse_position_munk = MunkToGame(mouse_position, height)

        move_command.dest_position = mouse_position

        ### Clear screen
        screen.fill(pygame.color.THECOLORS["black"])
        
        ### Draw stuff
        space.debug_draw(draw_options)

        if control_mode == "target_pos":
            position_control.debugInfoDraw( debug_info )
        elif control_mode == "auto":
            #behaviors[2].debugVisDraw(debug_info)
            #drawDebugCircle(debug_info, "green", 5, dest_point)
            vis_behaviors = ["keep_contact", "push_object", "spatial_path_planner", "obstacle_avoidance", "move_towards"]
            for b in behaviors:
                if b.name in vis_behaviors and b.name in active_behaviors:
                    b.debugVisDraw(debug_info)

        # draw debug info
        if show_debug:
            drawDebugInfo(screen, height, debug_info)

        # Info and flip screen
        screen.blit(font.render("fps: " + str(clock.get_fps()), 1, THECOLORS["white"]), (0,0))
        screen.blit(font.render("Mouse position (in world coordinates): " + str(mouse_position[0]) + "," + str(mouse_position[1]), 1, THECOLORS["darkgrey"]), (5,height - 35))
        screen.blit(font.render("Press ESC or Q to quit", 1, THECOLORS["darkgrey"]), (5,height - 20))
        
        pygame.display.flip()
        
        ### Update physics
        fps = 60
        dt = 1./fps
        space.step(dt)
        
        clock.tick(fps)

if __name__ == '__main__':
    sys.exit(main())

