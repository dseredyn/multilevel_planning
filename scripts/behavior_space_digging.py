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

class RoutesGraph:
    def __init__(self):
        self.routes = {}
        self.connections = {}

    def addRoute(self, route):
        v1_idx = route[0]
        v2_idx = route[-1]
        if v1_idx < v2_idx:
            route_id = (v1_idx, v2_idx)
        else:
            route_id = (v2_idx, v1_idx)
            route.reverse()
        if not route_id in self.routes:
            self.routes[route_id] = route

        if not v1_idx in self.connections:
            self.connections[v1_idx] = []
        self.connections[v1_idx].append( v2_idx )

        if not v2_idx in self.connections:
            self.connections[v2_idx] = []
        self.connections[v2_idx].append( v1_idx )

    def generate(self, edges_dict):
        route_id = 0
        for v in edges_dict:
            if len(edges_dict[v]) > 2:
                for nv in edges_dict[v]:
                    route = self.getPath(v, nv, edges_dict)
                    self.addRoute(route)

    # v1 is a junction node
    # v2 is a neighbour of v1
    # this function searches for path from v1 through v2 to next junction
    # this function is used in calculateShortcuts
    def getPath(self, v1, v2, edges_dict):
        assert len(edges_dict[v1]) > 2
        path = [ v1 ]
        next_node = v2
        current_node = v1
        while len(edges_dict[next_node]) == 2:
            path.append( next_node )
            for v in edges_dict[next_node]:
                if current_node != v:
                    current_node = next_node
                    next_node = v
                    break
        path.append( next_node )
        return path

# This behavior uses perception data to generate spatial information about environment.
class BehaviorSpaceDigging:

    def __init__(self, models_list):
        self.name = 'space_digging'
        self.inputs = ['env_perception', 'robot_pose']#, 'digging_push']
        self.outputs = [ 'spatial_map2' ]

        self.info_perception_space = None
        self.info_robot_pose = None

        self.result = None
        self.models_list = models_list
        self.edges_dict = {}
        self.edges_lengths = {}

    def addEdge2(self, s1_idx, s2_idx, p1_idx, p2_idx):
        assert s1_idx != s2_idx
        if s1_idx < s2_idx:
            s1, s2 = s1_idx, s2_idx
        else:
            s1, s2 = s2_idx, s1_idx

        if p1_idx < p2_idx:
            p1, p2 = p1_idx, p2_idx
        else:
            p1, p2 = p2_idx, p1_idx

        self.edges2.add( (s1, s2, p1, p2) )

    def floodfillSearch(self, starting_point):
        start_idx = self.tri.find_simplex( np.asarray( [starting_point] ) )[0]
        open_set = set()
        closed_set = set()
        open_set.add( start_idx )

        min_size = 50.0
        squared_min_size = min_size**2
        opposite_v = [ (1,2), (0,2), (0,1) ]
        self.edges2 = set()

        while len(open_set) > 0:
            # for every element in open_set get all neighbours, and if the transition is possible,
            # add them to a new open set
            new_open_set = set()
            for s_idx in open_set:
                for v_idx in range(3):
                    n_idx = self.tri.neighbors[s_idx, v_idx]
                    if n_idx == -1 or n_idx in closed_set or n_idx in open_set or n_idx in new_open_set:
                        continue
                    # get size of the edge
                    a_idx = self.tri.simplices[s_idx,opposite_v[v_idx][0]]
                    b_idx = self.tri.simplices[s_idx,opposite_v[v_idx][1]]
                    edge_size = np.sum(np.square(np.subtract(self.tri.points[a_idx,:], self.tri.points[b_idx,:])))
                    if edge_size > squared_min_size:
                        # add the neighbour to new open set
                        new_open_set.add( n_idx )
                    else:
                        # save the edge
                        self.addEdge2( s_idx, n_idx, a_idx, b_idx )
            closed_set = closed_set.union(open_set)
            open_set = new_open_set

    def plotBorder(self):
        for s1_idx, s2_idx, p1_idx, p2_idx in self.edges2:
            a = self.tri.points[p1_idx,:]
            b = self.tri.points[p2_idx,:]
            if self.points_labels_s[p1_idx] == 0 and self.points_labels_s[p2_idx] == 0:
                color = "r"
            elif self.points_labels_s[p1_idx] == 1 and self.points_labels_s[p2_idx] == 1:
                color = "g"
            else:
                color = "y"
            plt.plot( (a[0], b[0]), (a[1], b[1]), color )

    def update(self, information):
        self.info_perception_space = None
        self.info_robot_pose = None
#        self.info_digging_push = None
        for info in information:
            if info.type == 'env_perception':
                self.info_perception_space = info
                info.read = True
            if info.type == 'robot_pose':
                self.info_robot_pose = info
                info.read = True
#            if info.type == 'digging_push':
#                self.info_digging_push = info
#                info.read = True

        if self.info_perception_space is None or self.info_robot_pose is None:
            self.valid = False
        else:
            self.valid = True

        if not self.valid:
            return []

        # get the information
        point_inside = self.info_robot_pose.position

        if not self.result is None:
            # we assume that the map is static, so all calculations are made only once
            return [ ]#self.result ]

        self.result = InformationSpatialMap()

        #
        # use only points of static obstacles
        #
        points = []
        points_labels = []
        for model_idx in range(len(self.models_list)):
            model = self.models_list[model_idx]
            if model.name == "robot":
                continue
            if not model.static:
                continue
            for pt, n in model.surface:
                points.append( Vec2d(pt).rotated(model.body.angle) + model.body.position )
                points_labels.append( -1 )

        self.points_labels_s = np.asarray(points_labels, dtype=int)
        self.points_s = np.asarray(points)

        #
        # calculate Delaunay triangulation
        #
        self.tri = scipy.spatial.Delaunay(self.points_s)

        self.occluded_simplices = {}
        # link triangles to moveable obstacles
        for model_idx in range(len(self.models_list)):
            model = self.models_list[model_idx]
            if model.name == "robot":
                continue
            if model.static:
                continue
            for pt, n in model.surface:
                pt_W = Vec2d(pt).rotated(model.body.angle) + model.body.position
                s_idx = self.tri.find_simplex( np.asarray( [pt_W] ) )[0]
                if not s_idx in self.occluded_simplices:
                    self.occluded_simplices[s_idx] = set()
                self.occluded_simplices[s_idx].add( model_idx )

        if False:
            to_remove = set()
            remove_sequence = []
            while True:
                points = []
                points_labels = []
                for model_idx in range(len(self.models_list)):
                    if model_idx in to_remove:
                        continue
                    model = self.models_list[model_idx]
                    if model.name == "robot":
                        continue
                    for pt, n in model.surface:
                        points.append( Vec2d(pt).rotated(model.body.angle) + model.body.position )
                        if model.static:
                            points_labels.append( -1 )
                        else:
                            points_labels.append( model_idx )

                self.points_labels_s = np.asarray(points_labels, dtype=int)
                self.points_s = np.asarray(points)

                #
                # calculate Delaunay triangulation
                #
                self.tri = scipy.spatial.Delaunay(self.points_s)

                self.floodfillSearch( point_inside )
                removed = set()
                for s1_idx, s2_idx, p1_idx, p2_idx in self.edges2:
                    a = self.tri.points[p1_idx,:]
                    b = self.tri.points[p2_idx,:]
                    if self.points_labels_s[p1_idx] != -1:
                        to_remove.add( self.points_labels_s[p1_idx] )
                        removed.add( self.points_labels_s[p1_idx] )
                    if self.points_labels_s[p2_idx] != -1:
                        to_remove.add( self.points_labels_s[p2_idx] )
                        removed.add( self.points_labels_s[p2_idx] )
                if len(removed) == 0:
                    break
                remove_sequence.append( removed )
                break
            print remove_sequence

        # calculate centers for each simplex:
        # * mean
        # * circumcenter
        self.circumcenters = np.zeros( (self.tri.simplices.shape[0], 2) )
        self.meancenters = np.zeros( (self.tri.simplices.shape[0], 2) )
        for s_idx in range(self.tri.simplices.shape[0]):
            simplex_points = np.zeros( (3,2) )
            simplex_points[0,:] = self.tri.points[self.tri.simplices[s_idx,0], :]
            simplex_points[1,:] = self.tri.points[self.tri.simplices[s_idx,1], :]
            simplex_points[2,:] = self.tri.points[self.tri.simplices[s_idx,2], :]
            self.circumcenters[s_idx,:] = calculateCircumcenter(simplex_points)
            self.meancenters[s_idx,:] = np.multiply( simplex_points[0,:] + simplex_points[1,:] + simplex_points[2,:], 1.0/3.0 )

        # for each simplex calculate radius of circumcenter circle
        self.circumradius = np.zeros( (self.tri.simplices.shape[0], ) )
        for s_idx in range(self.tri.simplices.shape[0]):
            self.circumradius[s_idx] = math.sqrt(np.sum(np.square(self.tri.points[self.tri.simplices[s_idx,0], :] - self.circumcenters[s_idx,:])))

        # create graph on the mesh by connecting faces with passage greater than specified
#        min_size = 10.0
#        squared_min_size = min_size**2
#        opposite_v = [ (1,2), (0,2), (0,1) ]
#        for s_idx in range(self.tri.neighbors.shape[0]):
#            for v_idx in range(3):
#                n_idx = self.tri.neighbors[s_idx,v_idx]
#                if n_idx != -1:
#                    a_idx = self.tri.simplices[s_idx,opposite_v[v_idx][0]]
#                    b_idx = self.tri.simplices[s_idx,opposite_v[v_idx][1]]
#                    size = np.sum(np.square(np.subtract(self.tri.points[a_idx,:], self.tri.points[b_idx,:])))
#                    if size > squared_min_size:
#                        self.addEdge( s_idx, n_idx )

        self.routes_graph = RoutesGraph()
        self.routes_graph.generate( self.edges_dict )

        # calculate second Delaunay triangulation that takes points from Voronoi diagram
        #border_voronoi = np.concatenate( (border_s, self.circumcenters), axis=0)
#        border_voronoi = np.concatenate( (points_static_s, self.circumcenters), axis=0)
#        self.tri_vor = scipy.spatial.Delaunay(border_voronoi)

        return [ ]#self.result]

    def plotTriangulation(self):
        scipy.spatial.delaunay_plot_2d(self.tri)

    def plotTriangulationVor(self):
        scipy.spatial.delaunay_plot_2d(self.result.tri_vor)

    def plotOccludedSimplices(self):
        for s_idx in self.occluded_simplices:
            #occluded_simplices[s_idx]
            plt.plot( (self.meancenters[s_idx,0]), (self.meancenters[s_idx,1]), 'ro')

        for model_idx in range(len(self.models_list)):
            model = self.models_list[model_idx]
            if model.name == "robot":
                continue
            if model.static:
                continue
            for pt, n in model.surface:
                pt_W = Vec2d(pt).rotated(model.body.angle) + model.body.position
                plt.plot( (pt_W[0]), (pt_W[1]), 'yo')

    def plotGraph(self):
        edges = set()
        for v1 in self.edges_dict:
            for v2 in self.edges_dict[v1]:
                if v1 < v2:
                    edges.add( (v1, v2) )
                else:
                    edges.add( (v2, v1) )
        for e in edges:
            plt.plot( (self.result.circumcenters[e[0],0], self.result.circumcenters[e[1],0]), (self.result.circumcenters[e[0],1], self.result.circumcenters[e[1],1]), 'r')#, 'ro')

        print "edges:", len(edges)

        return
        for s_idx in range(0, self.result.circumcenters.shape[0], 1):
            cx = self.result.circumcenters[s_idx,0]
            cy = self.result.circumcenters[s_idx,1]
            r = self.result.circumradius[s_idx]
            circle_points_x = []
            circle_points_y = []
            for angle in np.linspace(0.0, math.pi*2.0, 20, endpoint=True):
                circle_points_x.append( math.cos(angle)*r + cx )
                circle_points_y.append( math.sin(angle)*r + cy )
            plt.plot( circle_points_x, circle_points_y)#, 'ro')

#        plt.show()

    def plotPath(self, path):
        for i in range(len(path)-1):
            print self.result.circumcenters[path[i],:], self.result.circumcenters[path[i+1],:]
            plt.plot( (self.result.circumcenters[path[i],0], self.result.circumcenters[path[i+1],0]), (self.result.circumcenters[path[i],1], self.result.circumcenters[path[i+1],1]), "ro" )
#        plt.show()

#    def addEdge(self, e1, e2):
#        if not e1 in self.edges_dict:
#            self.edges_dict[e1] = set()
#        if not e2 in self.edges_dict:
#            self.edges_dict[e2] = set()
#
#        self.edges_dict[e1].add( e2 )
#        self.edges_dict[e2].add( e1 )
#
#        self.edges_lengths[ (e1,e2) ] = math.sqrt(np.sum(np.square(np.subtract( self.circumcenters[e1,:], self.circumcenters[e2,:] ))))
#        self.edges_lengths[ (e2,e1) ] = self.edges_lengths[ (e1,e2) ]

