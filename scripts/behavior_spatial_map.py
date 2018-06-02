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

class DistanceSimplexObstacle:
    def __init__(self, s_idx, v1_idx, v2_idx):
        self.type = 0
        self.s_idx = s_idx
        self.v1_idx = v1_idx
        self.v2_idx = v2_idx

class DistanceSimplexCircumcenter:
    def __init__(self, s1_idx, p_idx, s2_idx):
        self.type = 1
        self.s1_idx = s1_idx
        self.s2_idx = s2_idx
        self.p_idx = p_idx

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
class BehaviorSpatialMapGeneration:

    def __init__(self, models_list):
        self.name = 'spatial_map_generation'
        self.inputs = ['env_perception', 'robot_pose']#, 'spatial_map_range']
        self.outputs = [ 'spatial_map' ]

        self.info_perception_space = None
        self.info_robot_pose = None
        #self.info_spatial_map_range = None

        self.result = None
        self.models_list = models_list
        self.iteration = 0
        self.objects_state_dict = {}

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

#    def isInCollision(self, point):
#        start_idx = self.result.tri_obs.find_simplex( np.asarray( [starting_point] ) )[0]

    def floodfillSearch(self, starting_point):
        start_idx = self.result.tri_obs.find_simplex( np.asarray( [starting_point] ) )[0]
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
                    n_idx = self.result.tri_obs.neighbors[s_idx, v_idx]
                    if n_idx == -1 or n_idx in closed_set or n_idx in open_set or n_idx in new_open_set:
                        continue
                    # get size of the edge
                    a_idx = self.result.tri_obs.simplices[s_idx,opposite_v[v_idx][0]]
                    b_idx = self.result.tri_obs.simplices[s_idx,opposite_v[v_idx][1]]
                    edge_size = np.sum(np.square(np.subtract(self.result.tri_obs.points[a_idx,:], self.result.tri_obs.points[b_idx,:])))
                    if edge_size > squared_min_size:
                        # add the neighbour to new open set
                        new_open_set.add( n_idx )
                    else:
                        # save the edge
                        self.addEdge2( s_idx, n_idx, a_idx, b_idx )
            closed_set = closed_set.union(open_set)
            open_set = new_open_set

        self.triangles_meaning = np.zeros( (self.result.tri_obs.simplices.shape[0],), dtype=int)
        for s_idx in closed_set:
            self.triangles_meaning[s_idx] = 1

    def plotBorder(self):
        for s1_idx, s2_idx, p1_idx, p2_idx in self.edges2:
            a = self.result.tri_obs.points[p1_idx,:]
            b = self.result.tri_obs.points[p2_idx,:]
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
        #self.info_spatial_map_range = None
        for info in information:
            if info.type == 'env_perception':
                self.info_perception_space = info
                info.read = True
            if info.type == 'robot_pose':
                self.info_robot_pose = info
                info.read = True
            #if info.type == 'spatial_map_range':
            #    self.info_spatial_map_range = info
            #    info.read = True

        if self.info_perception_space is None or self.info_robot_pose is None:# or self.info_spatial_map_range is None:
            self.valid = False
        else:
            self.valid = True

        if not self.valid:
            return []

        # get the information
        space = self.info_perception_space.space

        point_inside = self.info_robot_pose.position

        #range_x = self.info_spatial_map_range.range_x
        #range_y = self.info_spatial_map_range.range_y
        #steps_x = self.info_spatial_map_range.steps_x
        #steps_y = self.info_spatial_map_range.steps_y

#        if self.iteration >= 10:
#            self.iteration = 0
#        else:
#            self.iteration += 1

#        if self.iteration != 0 and not self.result is None:
#            return [ self.result ]
            
#        if not self.result is None:
#            # we assume that the map is static, so all calculations are made only once
#            return [ self.result ]

        objects_state_dict = {}
        points = []
        points_labels = []
        for model in self.models_list:
            if model.name == "robot":
                continue
            for pt, n in model.surface:
                points.append( Vec2d(pt).rotated(model.body.angle) + model.body.position )
                objects_state_dict[model.name] = (model.body.position, model.body.angle)
                if model.static:
                    points_labels.append( 0 )
                else:
                    points_labels.append( 1 )

        environment_changed = False
        for name in objects_state_dict:
            if not name in self.objects_state_dict:
                environment_changed = True
                break
            state1 = objects_state_dict[name]
            state2 = self.objects_state_dict[name]
            if (state1[0]-state2[0]).get_length() > 5.0 or abs(wrapAnglePI(state1[1]-state2[1])) > math.pi/20.0:
                environment_changed = True
                break

        if not environment_changed and not self.result is None:
            return [ self.result ]

        if environment_changed:
            self.objects_state_dict = objects_state_dict
            print "environment_changed"

        self.points_labels_s = np.asarray(points_labels, dtype=int)
        self.points_s = np.asarray(points)
        #for p_idx in range(points_s.shape[0]):
        #    print points_s[p_idx,:]

        result = InformationSpatialMap()
        if self.result is None:
            result.spatial_map_id = 1
        else:
            result.spatial_map_id = self.result.spatial_map_id + 1

        #
        # calculate Delaunay triangulation
        #
        try:
            result.tri_obs = scipy.spatial.Delaunay(self.points_s)
        except:
            if self.result is None:
                return []
            else:
                return [ self.result ]

        #
        # calculate distance to obstacles
        #
        #self.result.dist_map = buildDistanceMap( space, range_x, steps_x, range_y, steps_y )

        #
        # calculate border of free space
        #
        #point_inside_img = (int((float(point_inside[0])-range_x[0])*steps_x/(range_x[1] - range_x[0])),
        #                int((float(point_inside[1])-range_y[0])*steps_y/(range_y[1] - range_y[0])))
        #self.border = getFreeSpaceBorder( self.result.dist_map, point_inside_img )
        ## convert coordinates from image to world
        #for i in range(len(self.border)):
        #    self.border[i] = Vec2d( float(self.border[i][0])*(range_x[1] - range_x[0])/steps_x + range_x[0],
        #                            float(self.border[i][1])*(range_y[1] - range_y[0])/steps_y + range_y[0] )
        #print "len(BehaviorSpatialMap.border)", len(self.border)
        #border_s = np.asarray(self.border)

        #img_border = np.zeros( self.result.dist_map.shape )
        #for pt in self.border:
        #    img_border[pt[1], pt[0]] = 1
        #plt.imshow(img_border, interpolation='nearest')
        #plt.show()
        #for i in range(30):
        #    self.result.dist_map = np.subtract(self.result.dist_map, 2.0)
        #    self.result.dist_map = np.clip(self.result.dist_map, 0, None)
        #    cv2.imwrite('/home/dseredyn/Obrazy/test/dist_' + str(i) + '.png', normalizeImage(self.result.dist_map))


        #
        # calculate Delaunay triangulation
        #
        #self.result.tri = scipy.spatial.Delaunay(border_s)

#        points_static = []
#        for model in self.models_list:
#            if model.name == "robot":
#                continue
#            for pt, n in model.surface:
#                if model.static:
#                    points_static.append( Vec2d(pt).rotated(model.body.angle) + model.body.position )
#        points_static_s = np.asarray(points_static)

        #
        # calculate Delaunay triangulation
        #
        #result.tri = scipy.spatial.Delaunay(points_static_s)

        result.tri = result.tri_obs
        points_static_s = self.points_s

        # calculate centers for each simplex:
        # * mean
        # * circumcenter
        result.circumcenters = np.zeros( (result.tri.simplices.shape[0], 2) )
        result.meancenters = np.zeros( (result.tri.simplices.shape[0], 2) )
        for s_idx in range(result.tri.simplices.shape[0]):
            simplex_points = np.zeros( (3,2) )
            simplex_points[0,:] = result.tri.points[result.tri.simplices[s_idx,0], :]
            simplex_points[1,:] = result.tri.points[result.tri.simplices[s_idx,1], :]
            simplex_points[2,:] = result.tri.points[result.tri.simplices[s_idx,2], :]
            result.circumcenters[s_idx,:] = calculateCircumcenter(simplex_points)
            result.meancenters[s_idx,:] = np.multiply( simplex_points[0,:] + simplex_points[1,:] + simplex_points[2,:], 1.0/3.0 )

        # for each simplex calculate radius of circumcenter circle
        result.circumradius = np.zeros( (result.tri.simplices.shape[0], ) )
        for s_idx in range(result.tri.simplices.shape[0]):
            result.circumradius[s_idx] = math.sqrt(np.sum(np.square(result.tri.points[result.tri.simplices[s_idx,0], :] - result.circumcenters[s_idx,:])))

        #start_idx = result.tri.find_simplex( np.asarray( [point_inside] ) )
        #print "robot is in face", start_idx

        # create graph on the mesh by connecting faces with passage greater than specified
        min_size = 10.0
        squared_min_size = min_size**2
        opposite_v = [ (1,2), (0,2), (0,1) ]
        for s_idx in range(result.tri.neighbors.shape[0]):
            for v_idx in range(3):
                n_idx = result.tri.neighbors[s_idx,v_idx]
                if n_idx != -1:
                    a_idx = result.tri.simplices[s_idx,opposite_v[v_idx][0]]
                    b_idx = result.tri.simplices[s_idx,opposite_v[v_idx][1]]
                    size = np.sum(np.square(np.subtract(result.tri.points[a_idx,:], result.tri.points[b_idx,:])))
                    if size > squared_min_size:
                        self.addEdge( s_idx, n_idx, result )

#        self.routes_graph = RoutesGraph()
#        self.routes_graph.generate( result.edges_dict )
#
#        # calculate curvature
#        for route_id in self.routes_graph.routes:
#            route = self.routes_graph.routes[route_id]
#            for idx in range(len(route)-2):
#                i1 = route[idx]
#                i2 = route[idx+1]
#                i3 = route[idx+2]
#                c1 = result.circumcenters[i1,:]
#                c2 = result.circumcenters[i2,:]
#                c3 = result.circumcenters[i3,:]
#                v1 = array2Vec2d(c1)-array2Vec2d(c2)
#                v2 = array2Vec2d(c2)-array2Vec2d(c3)
#                angle = v1.get_angle_between(v2)
#
#        # calculate width
#        for route_id in self.routes_graph.routes:
#            route = self.routes_graph.routes[route_id]
#            for idx in range(len(route)-1):
#                i1 = route[idx]
#                i2 = route[idx+1]
#                pt_idx = self.getCommonEdge(i1, i2)
#                p1 = array2Vec2d(result.tri.points[pt_idx[0],:])
#                p2 = array2Vec2d(result.tri.points[pt_idx[1],:])
#                size = (p1-p2).get_length()

        # calculate second Delaunay triangulation that takes points from Voronoi diagram
        #border_voronoi = np.concatenate( (border_s, result.circumcenters), axis=0)
        border_voronoi = np.concatenate( (points_static_s, result.circumcenters), axis=0)
        try:
            #result.tri_vor = scipy.spatial.Delaunay(border_voronoi)
            pass
        except:
            if self.result is None:
                return []
            else:
                return [ self.result ]

        if False:
            # calculate distance representation

            # for every circumcenter, create triangles that contain circumcenter and neigbouring points of obstacle
            self.distance_triangles = []
            for s_idx in range(result.tri.simplices.shape[0]):
                for v1_idx in range(3):
                    v2_idx = (v1_idx+1)%3
                    a_idx = result.tri.simplices[s_idx,v1_idx]
                    b_idx = result.tri.simplices[s_idx,v2_idx]
                    size = np.sum(np.square(np.subtract(result.tri.points[a_idx,:], result.tri.points[b_idx,:])))
                    if size <= squared_min_size:
                        self.distance_triangles.append( DistanceSimplexObstacle(s_idx, v1_idx, v2_idx) )

            # for every two neighbouring circumcenters, create triangles that contain common vertex of both circumcenters
            edges = set()
            for s1_idx in result.edges_dict:
                for s2_idx in result.edges_dict[s1_idx]:
                    if s1_idx < s2_idx:
                        edges.add( (s1_idx, s2_idx) )
                    else:
                        edges.add( (s2_idx, s1_idx) )

            for s1_idx, s2_idx in edges:
                for v1_idx in range(3):
                    for v2_idx in range(3):
                        if result.tri.simplices[s1_idx,v1_idx] == result.tri.simplices[s2_idx,v2_idx]:
                            self.distance_triangles.append( DistanceSimplexCircumcenter(s1_idx, result.tri.simplices[s1_idx,v1_idx], s2_idx) )
        self.result = result
        self.floodfillSearch(point_inside)
        self.result.triangles_meaning = self.triangles_meaning
        return [self.result]

    def getCommonEdge(self, s1_idx, s2_idx):
        s1 = self.result.tri.simplices[s1_idx,:]
        s2 = self.result.tri.simplices[s2_idx,:]
        pt_idx = []
        for v1_idx in range(3):
            for v2_idx in range(3):
                if s1[v1_idx] == s2[v2_idx]:
                    pt_idx.append( s1[v1_idx] )
        assert len(pt_idx) == 2
        return pt_idx

    def plotDistanceGraph(self):
        for tr in self.distance_triangles:
            if tr.type == 0:
                a_idx = self.result.tri.simplices[tr.s_idx,tr.v1_idx]
                b_idx = self.result.tri.simplices[tr.s_idx,tr.v2_idx]
                a = self.result.tri.points[a_idx,:]
                b = self.result.tri.points[b_idx,:]
                c = self.result.circumcenters[tr.s_idx,:]
                plt.plot( (a[0], b[0], c[0], a[0]), (a[1], b[1], c[1], a[1]), "r" )
            elif tr.type == 1:
                a = self.result.circumcenters[tr.s1_idx,:]
                b = self.result.circumcenters[tr.s2_idx,:]
                c = self.result.tri.points[tr.p_idx,:]
                plt.plot( (a[0], b[0], c[0], a[0]), (a[1], b[1], c[1], a[1]), "g" )

    def plotTriangulation(self):
        scipy.spatial.delaunay_plot_2d(self.result.tri)

    def plotTriangulationVor(self):
        scipy.spatial.delaunay_plot_2d(self.result.tri_vor)

    def plotTriangulationObs(self):
        scipy.spatial.delaunay_plot_2d(self.result.tri_obs)

    def plotGraphObs(self):
        edges = set()
        for v1 in self.result.edges_dict:
            for v2 in self.result.edges_dict[v1]:
                if v1 < v2:
                    edges.add( (v1, v2) )
                else:
                    edges.add( (v2, v1) )
        for e in edges:
            plt.plot( (self.result.circumcenters[e[0],0], self.result.circumcenters[e[1],0]), (self.result.circumcenters[e[0],1], self.result.circumcenters[e[1],1]), 'r')#, 'ro')

    def plotGraph(self):
        edges = set()
        for v1 in self.result.edges_dict:
            for v2 in self.result.edges_dict[v1]:
                if v1 < v2:
                    edges.add( (v1, v2) )
                else:
                    edges.add( (v2, v1) )
        for e in edges:
            plt.plot( (self.result.circumcenters[e[0],0], self.result.circumcenters[e[1],0]), (self.result.circumcenters[e[0],1], self.result.circumcenters[e[1],1]), 'r')#, 'ro')

#        for pt in self.border:
#            plt.plot( pt[0], pt[1] , 'bo')
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

    def plotDistanceMap(self):
        plt.imshow(self.result.dist_map, origin='lower')

    def plotPath(self, path):
        for i in range(len(path)-1):
            print self.result.circumcenters[path[i],:], self.result.circumcenters[path[i+1],:]
            plt.plot( (self.result.circumcenters[path[i],0], self.result.circumcenters[path[i+1],0]), (self.result.circumcenters[path[i],1], self.result.circumcenters[path[i+1],1]), "ro" )
#        plt.show()

    def addEdge(self, e1, e2, info):
        if not e1 in info.edges_dict:
            info.edges_dict[e1] = set()
        if not e2 in info.edges_dict:
            info.edges_dict[e2] = set()

        info.edges_dict[e1].add( e2 )
        info.edges_dict[e2].add( e1 )

        info.edges_lengths[ (e1,e2) ] = math.sqrt(np.sum(np.square(np.subtract( info.circumcenters[e1,:], info.circumcenters[e2,:] ))))
        info.edges_lengths[ (e2,e1) ] = info.edges_lengths[ (e1,e2) ]

