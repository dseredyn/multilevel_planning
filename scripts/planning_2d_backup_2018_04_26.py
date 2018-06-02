#!/usr/bin/env python

import sys
import math
import numpy as np
import random

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

from task_routines import *
from information import *
from behavior_spatial_map import *

def forceControl(robot):
    keys = pygame.key.get_pressed()
    if (keys[K_UP]):
        robot.apply_impulse_at_local_point(Vec2d(0,100), (0,0))
    if (keys[K_DOWN]):
        robot.apply_impulse_at_local_point(Vec2d(0,-100), (0,0))
    if (keys[K_LEFT]):
        robot.apply_impulse_at_local_point(Vec2d(-100,0), (0,0))
    if (keys[K_RIGHT]):
        robot.apply_impulse_at_local_point(Vec2d(100,0), (0,0))

def main():
    sys.setrecursionlimit(10000)

    if False:
        plot_data_x = []
        plot_data_y = []
        for angle in np.linspace(0.0, math.pi*2.0, 500):
            plot_data_x.append( angle )
            plot_data_y.append( stats.norm.pdf(angle, math.pi/4.0, math.pi/16.0) + stats.norm.pdf(angle, math.pi/4.0+math.pi, math.pi/16.0) )

        plt.plot( plot_data_x, plot_data_y )
        plt.text(math.pi/4.0, stats.norm.pdf(0, 0, math.pi/16.0), "A")
        plt.text(math.pi/4.0+math.pi, stats.norm.pdf(0, 0, math.pi/16.0), "B")
        plt.show()
        exit(0)

    width, height = 690,600

    ### PyGame init
    pygame.init()
    screen = pygame.display.set_mode((width,height)) 
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.SysFont("Arial", 16)
    
    ### Physics stuff
    space = pymunk.Space()   
#    space.gravity = 0,-1000
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # walls - the left-top-right walls
    static= [pymunk.Segment(space.static_body, (50, 50), (50, 550), 5)
                ,pymunk.Segment(space.static_body, (50, 550), (650, 550), 5)
                ,pymunk.Segment(space.static_body, (650, 550), (650, 50), 5)
                ,pymunk.Segment(space.static_body, (50, 50), (650, 50), 5)
#                ,pymunk.Segment(space.static_body, (323, 134), (566, 130), 5)
#                ,pymunk.Segment(space.static_body, (566, 130), (532, 367), 5)
                ]  
    
#    b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
#    static.append(pymunk.Circle(b2, 30))
#    b2.position = 300,400
    static.append(pymunk.Circle(space.static_body, 30, (300,400)))
    
    for s in static:
        s.friction = 1.
        s.group = 1
    space.add(static)

#    cab = ModelCabinet(200, 100, (200,200), 0.0)
    cab = ModelCabinet(200, 100, Vec2d(200,200), -0.05)
    cab.addToSpace(space)

    # robot
    rob = ModelRobot( Vec2d(100,100), 0.0 )
#    rob = ModelRobot( Vec2d(270,100), -math.pi/2.0 )
#    rob = ModelRobot( Vec2d(180,200), -math.pi/2.0 )
    #rob = ModelRobot( Vec2d(200,230), math.pi/2.0 )
    rob.addToSpace(space)

    #createSpatialModelForBody(rob.robot_body)

    #rob_position_map = Vec2d(int(rob.robot_body.position[0]*100.0/width), int(rob.robot_body.position[1]*100.0/height))
    space_map = SpaceMap()
    #space_map.build( space, rob_position_map, (0, width), 100, (0, height), 100 )
    space_map.build( space, rob.robot_body.position, (0, width), 100, (0, height), 100 )
    space_map.calculateShortcuts()

    #space_map.plotGraph()
    #plt.show()
    # wyznaczenie punktow kontaktu i odpowiadajacych im kierunkow dzialania sily
#    exit(0)

    #TODO:
    #space.point_query_nearest(mouse_pos, pymunk.inf, pymunk.ShapeFilter())

    # target for manual control
    control_mode = "target_pos"
    target = [rob.robot_body.position[0], rob.robot_body.position[1], 0.0]

    control_mode = "auto"
    #move_target = Vec2d(150,450)
    #move_target = Vec2d(200,230)

    print "rob.robot_body.position", rob.robot_body.position
    #print "move_target", move_target
    #move_target_map = Vec2d(int(move_target[0]*100.0/width), int(move_target[1]*100.0/height))

    #space_map.setTarget(move_target)
    #path = space_map.findShortestPath(rob_position_map, move_target_map)
    #path = space_map.findShortestPath(rob.robot_body.position, move_target)
    #print path
    #space_map.plotTriangulation()
    #space_map.plotGraph()
    #space_map.plotPath(path)
#    plt.plot( [rob_position_map[0]], [rob_position_map[1]], "o" )
#    plt.plot( [move_target_map[0]], [move_target_map[1]], "o" )
    #plt.show()
#    exit(0)

    plan_idx = 0
    plan = [    ("move_to", Vec2d(180,200)),
#    plan = [    #("move_to", Vec2d(150,450)),
                ("open_door", ) ]

    # provide information
    information = [
        InformationSpatialMapRange( (0, width), 100, (0, height), 100 ),        # range and resolution of spatial map
        InformationRobotPose( rob.robot_body.position, rob.robot_body.angle ),  # current pose of robot
        InformationPerceptionSpace( space ),                                    # percepted environment (exact, full perception)
        InformationDestinationGeom( Vec2d(180,200) ),                           # target point for motion
    ]

    behaviors = []

    b_spatial_map = BehaviorSpatialMap()
    new_inf = b_spatial_map.update( information )
    joinInformation( information, new_inf )

#    print len(information)
#    exit(0)

    first_behavior_iteration = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or \
                event.type == KEYDOWN and (event.key in [K_ESCAPE, K_q]):  
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                start_time = pygame.time.get_ticks()
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(screen, "arrows.png")
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                #TODO: remove
                end_time = pygame.time.get_ticks()
                diff = end_time - start_time
                power = max(min(diff, 1000), 10) * 1.5
                impulse = power * Vec2d(1,0)

        debug_info = []

        # manual control
        if control_mode == "force":
            forceControl(rob.robot_body)
        elif control_mode == "target_pos":
            keys = pygame.key.get_pressed()
            if (keys[K_UP]):
                target[0] += math.cos(target[2])
                target[1] += math.sin(target[2])
            if (keys[K_DOWN]):
                target[0] -= math.cos(target[2])
                target[1] -= math.sin(target[2])
            if (keys[K_LEFT]):
                target[2] += 0.05
            if (keys[K_RIGHT]):
                target[2] -= 0.05
            lin_stiffness = 1000.0
            lin_damping = 500.0
            rot_stiffness = 10000.0
            rot_damping = 8000.0
            rob.robot_body.force = lin_stiffness * (Vec2d(target[0], target[1]) - rob.robot_body.position) - lin_damping * rob.robot_body.velocity
            rob.robot_body.torque = rot_stiffness * wrapAnglePI( target[2] - rob.robot_body.angle ) - rot_damping * rob.robot_body.angular_velocity
        elif control_mode == "auto":
            if plan_idx >= len(plan):
                    lin_damping = 50.0
                    rot_damping = 800.0
                    rob.robot_body.force = -lin_damping * rob.robot_body.velocity
                    rob.robot_body.torque = -rot_damping * rob.robot_body.angular_velocity
            elif plan[plan_idx][0] == "move_to":
                if (rob.robot_body.position-plan[plan_idx][1]).get_length() < 10.0:
                    plan_idx += 1
                    first_behavior_iteration = True
                    print "plan_idx", plan_idx
                else:
                    if first_behavior_iteration:        # initialize behavior
                        space_map.setTarget(plan[plan_idx][1])
                        first_behavior_iteration = False

                        # make predictions and extend the plan
                        predicted_path = space_map.findShortestPath(rob.robot_body.position)
                        tp = TaskTightPassages(rob, predicted_path, space_map)

                    force, torque = space_map.getDrivingForce(None, rob, plan[plan_idx][1])
                    force *= 1000.0
                    torque *= 2000.0
                    lin_damping = 50.0 * 0.5
                    rot_damping = 800.0# * 0.5
                    rob.robot_body.force = force - lin_damping * rob.robot_body.velocity
                    rob.robot_body.torque = torque - rot_damping * rob.robot_body.angular_velocity
            elif plan[plan_idx][0] == "open_door":
                if first_behavior_iteration:
                    first_behavior_iteration = False
                    door_instance_info = cab.getLeftDoorInfo()
                    clear_space = ClearSpace()
                    clear_space.addClearPoint( (Vec2d(200,250),50) )
                    #clear_space.getOperations(space)
                    exp_door = ExperienceDoor()
                    target_open_angle = exp_door.getDestPosition(clear_space, door_instance_info)
                door_instance_info = cab.getLeftDoorInfo()
                if abs(door_instance_info.open_angle - target_open_angle) < 0.2:
                    plan_idx += 1
                    first_behavior_iteration = True
                    print "plan_idx", plan_idx
                push = exp_door.getPush(target_open_angle, door_instance_info, debug_info)
                #exp_door.predictLeftDoorMovement(target_open_angle, door_instance_info, debug_info)
                #TODO: pushing

                #push_loc = space_map.pushLocations( push, rob )

                force, torque = space_map.getPushForce(push, rob, debug_info)
                lin_damping = 50.0
                rot_damping = 800.0
                rob.robot_body.force = 500.0*force - lin_damping * rob.robot_body.velocity
                rob.robot_body.torque = 500.0*torque - rot_damping * rob.robot_body.angular_velocity

                #loc = space_map.pushLocations(push, rob)
                #print loc
                #print push_loc[0][0], push_loc[0][1]
                #rob.debugVisPose(debug_info, push_loc[0][0], push_loc[0][1])
#        print "plan_idx", plan_idx
        #cab.debugVis(debug_info)

        rob.debugVisQhull(debug_info)
        rob.debugVisPushing(debug_info)

        mouse_position = pymunk.pygame_util.from_pygame( Vec2d(pygame.mouse.get_pos()), screen )
        mouse_position_munk = MunkToGame(mouse_position, height)

        ### Clear screen
        screen.fill(pygame.color.THECOLORS["black"])
        
        ### Draw stuff
        space.debug_draw(draw_options)

        # bounding circle for the robot
        #pygame.draw.circle(screen, pygame.color.THECOLORS["yellow"], MunkToGame(rob.robot_body.position.int_tuple, height), 42, 1)

        # draw target
        if control_mode == "target_pos":
            col_g = pygame.color.THECOLORS["green"]
            pygame.draw.circle(screen, col_g, MunkToGame( (int(target[0]), int(target[1])), height ), 10, 1)
            pygame.draw.line(screen, col_g, MunkToGame( (int(target[0]), int(target[1])), height ), MunkToGame( (int(target[0] + 15*math.cos(target[2])), int(target[1] + 15*math.sin(target[2]))), height ), 1)

        # draw debug info
        drawDebugInfo(screen, height, debug_info)

        # Power meter
        if pygame.mouse.get_pressed()[0]:
            current_time = pygame.time.get_ticks()
            diff = current_time - start_time
            power = max(min(diff, 1000), 10)
            h = power / 2
            pygame.draw.line(screen, pygame.color.THECOLORS["red"], (30,550), (30,550-h), 10)
                
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
