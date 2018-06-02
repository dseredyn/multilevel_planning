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
class BehaviorSpatialPathPlanner:
    def __init__(self):
        # common variables for all behaviors
        self.name = "spatial_path_planner"
        self.inputs = ['robot_pose', 'destination_geom', 'spatial_map', 'stuck']
        self.outputs = [ 'path' ]

        # input variables
        self.info_robot_pose = None
        self.info_destination_geom = None
        self.info_spatial_map = None
        self.info_stuck = None
        self.path = None

        # state variables
        self.target_idx = None
        self.updated_target = False
        self.ignored_nodes = set()

        self.debug_info = None
        self.spatial_map_id = None

    def update(self, information):
        self.info_robot_pose = None
        self.info_destination_geom = None
        self.info_spatial_map = None
        self.info_stuck = None
        for info in information:
            if info.type == 'robot_pose':
                self.info_robot_pose = info
                info.read = True
            elif info.type == 'destination_geom':
                self.info_destination_geom = info
                info.read = True
            elif info.type == 'spatial_map':
                self.info_spatial_map = info
                info.read = True
            elif info.type == 'stuck':
                self.info_stuck = info
                info.read = True

        if self.info_robot_pose is None or self.info_destination_geom is None or self.info_spatial_map is None:
            self.valid = False
        else:
            self.valid = True

        if not self.valid:
            return []

        self.setTarget( self.info_destination_geom.point )

        if self.spatial_map_id != self.info_spatial_map.spatial_map_id:
            map_updated = True
            self.spatial_map_id = copy.copy(self.info_spatial_map.spatial_map_id)
        else:
            map_updated = False

        #print self.updated_target
        if self.updated_target or map_updated:
            self.dist, self.prev = self.dijkstra(self.target_idx)
            self.path = self.findShortestPath( self.info_robot_pose.position )
            self.updated_target = False
            self.ignored_nodes = set()
            print "BehaviorSpatialPathPlanner: new path (target changed)"

        if not self.info_stuck is None:
            print "BehaviorSpatialPathPlanner: new path (stuck)"
            # we need to choose another path
            # cut the graph at current position
            current_s_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [self.info_robot_pose.position] ) )[0]
            #print "path", self.path
            #print "current_s_idx", current_s_idx
            next_s_idx = None
            for p_idx in range(len(self.path)-1):
                if self.path[p_idx] == current_s_idx:
                    next_s_idx = self.path[p_idx+1]
                    break
            #print "next_s_idx", next_s_idx
            if not next_s_idx is None:
                self.ignored_nodes.add( next_s_idx )
                self.dist, self.prev = self.dijkstra(self.target_idx)
                self.path = self.findShortestPath( self.info_robot_pose.position )
            #print "new path", self.path

        info_path = InformationPathGeom(self.path)
        info_path.prev = self.prev
        info_path.autoremove = True
        info_path.meancenters = self.info_spatial_map.meancenters
        info_path.circumcenters = self.info_spatial_map.circumcenters
        info_path.tri = self.info_spatial_map.tri

        self.debug_info = []
        for p_idx in range(len(self.path)):
            s_idx = self.path[p_idx]
            s = self.info_spatial_map.tri.simplices[s_idx,:]
            p1 = array2Vec2d( self.info_spatial_map.tri.points[s[0],:] )
            p2 = array2Vec2d( self.info_spatial_map.tri.points[s[1],:] )
            p3 = array2Vec2d( self.info_spatial_map.tri.points[s[2],:] )

            self.debug_info.append( ("line", "blue", p1, p2) )
            self.debug_info.append( ("line", "blue", p2, p3) )
            self.debug_info.append( ("line", "blue", p3, p1) )

        p4 = self.info_destination_geom.point
        self.debug_info.append( ("circle", "blue", p4, 20) )

        if not self.info_destination_geom.angle is None:
            p5 = p4 + Vec2d(25.0, 0).rotated(self.info_destination_geom.angle)
            self.debug_info.append( ("vector", "blue", p4, p5) )

        return [ info_path ]

    # calculates length of a path
    def getPathLength(self, path):
        length = 0.0
        for i in range(len(path)-1):
            length += self.info_spatial_map.edges_lengths[ (path[i], path[i+1]) ]

    def dijkstra(self, node_idx):
        Q = set()
        #print "len(self.info_spatial_map.edges_dict)", len(self.info_spatial_map.edges_dict)
        #print "self.info_spatial_map.tri.simplices.shape[0]", self.info_spatial_map.tri.simplices.shape[0]
        #print "dijkstra node_idx", node_idx
        dist = np.full( (self.info_spatial_map.tri.simplices.shape[0], ), float("inf") )     # Unknown distance from source to v
        prev = np.full( (self.info_spatial_map.tri.simplices.shape[0], ), -1, dtype=int )    # Previous node in optimal path from source
        for v in self.info_spatial_map.edges_dict:               # Initialization
            if v in self.ignored_nodes:
                continue
            Q.add(v)                            # All nodes initially in Q (unvisited nodes)
        dist[node_idx] = 0.0                    # Distance from node_idx to node_idx
        while bool(Q):                          # while not empty
            min_dist = float("inf")
            u = None
            for v in Q:
                if dist[v] < min_dist:          # Node with the least distance will be selected first
                    min_dist = dist[v]
                    u = v
            if u == None:
                break
            Q.remove(u)                         # remove u from Q 
            for v in self.info_spatial_map.edges_dict[u]:
                if not v in Q:                  # where v is still in Q.
                    continue
                alt = dist[u] + self.info_spatial_map.edges_lengths[ (u,v) ]
                if alt < dist[v]:               # A shorter path to v has been found
                    dist[v] = alt
                    prev[v] = u
        return dist, prev

    def findShortestPath(self, pt_source):
        source_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [pt_source] ) )[0]
#        target_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [pt_target] ) )[0]
#        print "source", source_idx
#        print "target", target_idx
#        dist, prev = self.dijkstra(target_idx)
        S  = []
        u = source_idx
        while self.prev[u] != -1:   # Construct the shortest path with a stack S
            S.append(u)             # Push the vertex onto the stack
            u = self.prev[u]        # Traverse from target to source
        S.append(u)                 # Push the source onto the stack
        return S

    def setTarget(self, pt_target):
        target_idx = self.info_spatial_map.tri.find_simplex( np.asarray( [pt_target] ) )[0]
        if target_idx != self.target_idx:
            self.target_idx = target_idx
            self.updated_target = True

    def debugVisDraw(self, debug_info):
        if not self.debug_info is None and not debug_info is None:
            debug_info += self.debug_info
        return

        if not self.path is None and not self.info_spatial_map is None and not debug_info is None:
            for p_idx in range(len(self.path)):
                s_idx = self.path[p_idx]
                #n_idx = self.path[p_idx+1]
                s = self.info_spatial_map.tri.simplices[s_idx,:]
                p1 = array2Vec2d( self.info_spatial_map.tri.points[s[0],:] )
                p2 = array2Vec2d( self.info_spatial_map.tri.points[s[1],:] )
                p3 = array2Vec2d( self.info_spatial_map.tri.points[s[2],:] )

#                p1 = array2Vec2d( self.info_spatial_map.circumcenters[s_idx,:] )
#                p2 = array2Vec2d( self.info_spatial_map.circumcenters[n_idx,:] )
                debug_info.append( ("line", "blue", p1, p2) )
                debug_info.append( ("line", "blue", p2, p3) )
                debug_info.append( ("line", "blue", p3, p1) )

                p4 = self.info_destination_geom.point
                debug_info.append( ("circle", "blue", p4, 20) )

                if not self.info_destination_geom.angle is None:
                    p5 = p4 + Vec2d(25.0, 0).rotated(self.info_destination_geom.angle)
                    debug_info.append( ("vector", "blue", p4, p5) )


