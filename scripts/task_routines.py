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

class ClearSpace:
    def __init__(self):
        self.clear_points = []

    # every point is given by 2-D coordinate and sigma (for gaussian function)
    def addClearPoint(self, p):
        self.clear_points.append( p )

    def getOperations(self, space):
        query_list = []
        for cp in self.clear_points:
            sq = space.segment_query( cp[0], cp[0], cp[1]*2.0, pymunk.ShapeFilter() )
            query_list += sq

#        for q in query_list:
#            if not q.shape is None:

class PushInfo:
    def __init__(self, n, pos, score, radius):
        self.n = n
        self.pos = pos
        self.score = score
        self.radius = radius

# 'user-manual' for door
class ExperienceDoor:
    def __init__(self):
        pass

    def getDestPosition(self, clear_space, door_instance_info):
        # get random angles and choose the best
        min_penalty = float("inf")
        best_angle = None
        for i in range(20):
            open_angle = random.uniform(0,math.pi/2)
            angle = open_angle + door_instance_info.angle - door_instance_info.open_angle
            penalty = 0.0
            for pt, n, radius in door_instance_info.samples:
                st = pt.rotated(angle) + door_instance_info.position #(  s[0] * math.cos(angle) + s[1] * math.sin(angle),
                        #-s[0] * math.sin(angle) + s[1] * math.cos(angle) )
                for cp, cp_sigma in clear_space.clear_points:
                    dist = (cp-st).get_length()
                    #dist = math.sqrt( (cp_x-st[0])**2 + (cp_y-st[1])**2 )
                    penalty += stats.norm.pdf(dist, 0, cp_sigma)
            print angle, penalty
            if penalty < min_penalty:
                min_penalty = penalty
                best_angle = open_angle

        #print "best_angle", best_angle
        return best_angle

    def getPush(self, target_open_angle, door_instance_info, debug_info=None):
        vec_list = []
        target_angle = target_open_angle + door_instance_info.angle - door_instance_info.open_angle
        current_angle = door_instance_info.angle
        #print target_angle, current_angle
        for pt, n, radius in door_instance_info.samples:
            target_st = pt.rotated(target_angle)
            current_st = pt.rotated(current_angle)
            if (target_st - current_st).dot(n.rotated(current_angle)) < 0.0:
                vec_list.append( (current_st, n.rotated(current_angle), radius) )
        #print "door_instance_info.angle", door_instance_info.angle
        if debug_info != None:
            for v in vec_list:
                debug_info.append( ("vector", "red", v[0]+door_instance_info.position, v[0]+v[1]*50+door_instance_info.position) )

        result = []
        for pt, n, radius in vec_list:
            result.append( PushInfo(n, pt+door_instance_info.position, pt.get_length(), radius) )
#            ort = Vec2d(A[1], -A[0])
#            length = ort.get_length()
#            if length > 0:
#                ort = ort / length*100
#            else:
#                continue
#            if ort.dot(B-A) > 0:
#                result.append( (ort, A+door_instance_info.position, A.get_length()) )
#            else:
#                result.append( (-ort, A+door_instance_info.position, A.get_length()) )
            if debug_info != None:
                debug_info.append( ("vector", "green", result[-1].pos, result[-1].pos+result[-1].n*30.0) )

        return result

    def predictLeftDoorMovement(self, target_open_angle, door_instance_info, debug_info=None):
        door_info_copy = copy.copy(door_instance_info)
        open_angle_diff = target_open_angle - door_instance_info.open_angle
        steps = int( open_angle_diff / (math.pi/16.0) )
        steps = max(3, steps)
        for angle_add in np.linspace(0.0, open_angle_diff, steps, endpoint=False):
            door_info_copy.angle = door_instance_info.angle + angle_add
            door_info_copy.open_angle = door_instance_info.open_angle + angle_add
            push = self.getPush(target_open_angle, door_info_copy)
            for pu in push:
                debug_info.append( ("vector", "green", pu[1], pu[1]+pu[0]) )

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

class ModelCabinet:
    def __init__(self, width, depth, position, rotation):
        self.width = width
        self.depth = depth
        self.position = position
        self.rotation = rotation

    def addToSpace(self, space):
        line_width = 4.0

        p1 = Vec2d(-self.width/2.0, self.depth/2.0)
        p2 = Vec2d(-self.width/2.0, -self.depth/2.0)
        p3 = Vec2d(self.width/2.0, -self.depth/2.0)
        p4 = Vec2d(self.width/2.0, self.depth/2.0)
        p5 = Vec2d(-self.width/2.0, self.depth/2.0 + line_width*2)     # left door joint

        t1 = p1.rotated(self.rotation) + self.position #transform(p1, self.position, self.rotation)
        t2 = p2.rotated(self.rotation) + self.position #transform(p2, self.position, self.rotation)
        t3 = p3.rotated(self.rotation) + self.position #transform(p3, self.position, self.rotation)
        t4 = p4.rotated(self.rotation) + self.position #transform(p4, self.position, self.rotation)

        t5 = p5.rotated(self.rotation) + self.position #transform(p5, self.position, self.rotation)

        static= [pymunk.Segment(space.static_body, t1, t2, line_width)
                    ,pymunk.Segment(space.static_body, t2, t3, line_width)
                    ,pymunk.Segment(space.static_body, t3, t4, line_width)
                    ]  

        for s in static:
            s.friction = 1.
            s.group = 1
        space.add(static)

        # samples for door
        # each sample contains information about surface point, normal and maximum radius of contacting body
        self.door_samples = []
        for t in np.linspace(0.1, 0.9, 5, endpoint=True):
            self.door_samples.append( (Vec2d(self.width/2.0 * t, -line_width/2.0), Vec2d(0,-1), float("inf")) )

        radius_tab = [ 100, 50, 15, 5, 5 ]
        idx = 0
        for t in np.linspace(0.1, 0.9, 5, endpoint=True):
            self.door_samples.append( (Vec2d(self.width/2.0 * t, line_width/2.0), Vec2d(0,1), radius_tab[idx]) )
            idx += 1

        self.door_samples.append( (Vec2d(self.width/2.0*0.8 - 4*2.0, 4*4.0), Vec2d(0,1), float("inf")) )
        self.door_samples.append( (Vec2d(self.width/2.0*0.8 - 4*2.0, 4*4.0), Vec2d(0,-1), 5) )

        # polygon for left door
        p6 = Vec2d(0, -line_width/2.0)
        p7 = Vec2d(0, line_width/2.0)
        p8 = Vec2d(self.width/2.0-line_width, line_width/2.0)
        p9 = Vec2d(self.width/2.0-line_width, -line_width/2.0)

#        ph1 = Vec2d(self.width/2.0*0.9, line_width/2.0)
#        ph2 = Vec2d(self.width/2.0*0.9, line_width*4.0)
#        ph3 = Vec2d(self.width/2.0*0.9-line_width*4.0, line_width*4.0)
#        ph4 = Vec2d(self.width/2.0*0.9-line_width*4.0, line_width*3.0)
#        ph5 = Vec2d(self.width/2.0*0.9-line_width*1.0, line_width*3.0)
#        ph6 = Vec2d(self.width/2.0*0.9-line_width*1.0, line_width/2.0)

        self.fp = [p9,p8,p7,p6]
        mass = 100.0
        moment = pymunk.moment_for_poly(mass, self.fp)

        # left door
        self.l_door_body = pymunk.Body(mass, moment)
        self.l_door_body.position = t5
        self.l_door_body.angle = self.rotation
        l_door_shape = pymunk.Poly(self.l_door_body, self.fp, radius=1)
        l_door_shape.group = 1

        # handle
        left_h1_shape = pymunk.Segment(self.l_door_body, Vec2d(self.width/2.0*0.8, line_width*0.5), Vec2d(self.width/2.0*0.8, line_width*4.0), line_width)
        left_h1_shape.group = 1

        left_h2_shape = pymunk.Segment(self.l_door_body, Vec2d(self.width/2.0*0.8 - line_width*4.0, line_width*4.0), Vec2d(self.width/2.0*0.8, line_width*4.0), line_width)
        left_h2_shape.group = 1

        space.add(self.l_door_body, l_door_shape, left_h1_shape, left_h2_shape)

        j = pymunk.PivotJoint(self.l_door_body, space.static_body, self.l_door_body.position)
        s = pymunk.DampedRotarySpring(self.l_door_body, space.static_body, 0, 0, 90000)
        space.add(j, s)

        j2 = pymunk.RotaryLimitJoint(self.l_door_body, space.static_body, -self.rotation-math.pi/2.0, -self.rotation)
        space.add(j2)

    def debugVis(self, debug_info = None):
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

class ModelRobot:
    def __init__(self, position, angle):
        self.mass = 10
        self.radius = 20
        self.inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0,0))
        self.position = position
        self.angle = angle

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

def normalizeImage(img):
    v_min = np.min(img)
    v_max = np.max(img)
    result = np.subtract(img, v_min)
    result = np.multiply( result, 255.0/(v_max-v_min) )
    return result

class SpaceMap:
    def __init__(self):
        self.edges_dict = {}
        self.edges_lengths = {}

    def buildDistanceMap(self, space, range_x, steps_x, range_y, steps_y):
        dist_map = np.zeros( (steps_y, steps_x) )
        ix = 0
        for x in np.linspace(range_x[0], range_x[1], steps_x, endpoint=True):
            iy = 0
            for y in np.linspace(range_y[0], range_y[1], steps_y, endpoint=True):
                pq = space.point_query_nearest((x,y), pymunk.inf, pymunk.ShapeFilter(group=2))
                if pq.distance > 0:
                    dist_map[iy, ix] = pq.distance
                iy += 1
            ix += 1
        return dist_map

    def recursiveGetFreeSpaceBorder(self, dist_map, visited_map, start_point, border):
        neighbours = [ (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1, -1), (0,-1) ]
        to_visit = []
        for n in neighbours:
            x = start_point[0] + n[0]
            y = start_point[1] + n[1]

            if x < 0 or x >= dist_map.shape[1] or y < 0 or y >= dist_map.shape[0]:
                continue
            if visited_map[y,x] == 1:
                continue

            visited_map[y,x] = 1
            if dist_map[y,x]  <= 0.0001:
                border.append( (x,y) )
                continue
            to_visit.append( (x,y) )

        for pt in to_visit:
            self.recursiveGetFreeSpaceBorder( dist_map, visited_map, pt, border )

    def getFreeSpaceBorder(self, dist_map, start_point):
        border = []
        visited_map = np.zeros( dist_map.shape, dtype=int)

        self.recursiveGetFreeSpaceBorder(dist_map, visited_map, start_point, border)

        return border

    def build(self, space, point_inside, range_x, steps_x, range_y, steps_y):
        self.dist_map = self.buildDistanceMap( space, range_x, steps_x, range_y, steps_y )
        
        point_inside = (int((float(point_inside[0])-range_x[0])*steps_x/(range_x[1] - range_x[0])),
                        int((float(point_inside[1])-range_y[0])*steps_y/(range_y[1] - range_y[0])))
        self.border = self.getFreeSpaceBorder( self.dist_map, point_inside )
        # convert coordinates from image to world
        for i in range(len(self.border)):
            self.border[i] = Vec2d( float(self.border[i][0])*(range_x[1] - range_x[0])/steps_x + range_x[0],
                                    float(self.border[i][1])*(range_y[1] - range_y[0])/steps_y + range_y[0] )
        print "len(self.border)", len(self.border)
        #plt.imshow(self.border, interpolation='nearest')
        #plt.show()

        #img_border = np.zeros( self.dist_map.shape )
        #for pt in self.border:
        #    img_border[pt[1], pt[0]] = 1
        #plt.imshow(img_border, interpolation='nearest')
        #plt.show()
        #for i in range(30):
        #    self.dist_map = np.subtract(self.dist_map, 2.0)
        #    self.dist_map = np.clip(self.dist_map, 0, None)
        #    cv2.imwrite('/home/dseredyn/Obrazy/test/dist_' + str(i) + '.png', normalizeImage(self.dist_map))

        border_s = np.asarray(self.border)

        self.tri = scipy.spatial.Delaunay(border_s)

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

        #start_idx = self.tri.find_simplex( np.asarray( [point_inside] ) )
        #print "robot is in face", start_idx

        # create graph on the mesh by connecting faces with passage greater than specified
        min_size = 50.0
        squared_min_size = min_size**2
        opposite_v = [ (1,2), (0,2), (0,1) ]
        for s_idx in range(self.tri.neighbors.shape[0]):
            for v_idx in range(3):
                n_idx = self.tri.neighbors[s_idx,v_idx]
                if n_idx != -1:
                    a_idx = self.tri.simplices[s_idx,opposite_v[v_idx][0]]
                    b_idx = self.tri.simplices[s_idx,opposite_v[v_idx][1]]
                    size = np.sum(np.square(np.subtract(self.tri.points[a_idx,:], self.tri.points[b_idx,:])))
                    if size > squared_min_size:
                        self.addEdge( s_idx, n_idx )

        #TODO: robot chodzi po trojkatach i bierze pod uwage ciasnosc przejscia. W zaleznosci od
        # tej szerokosci nalezy uruchomic dodatkowe zachowania, takie jak zmiana orientacji lub ulozenia
        # ciala, lub nawet przeciskanie sie, czyli przygotowanie sie do mozliwego kontaktu ze srodowiskiem.
        # na tym etapie schodzimy na nizszy poziom abstrakcji.
        #scipy.spatial.delaunay_plot_2d(self.tri)
        #plt.show()

        #self.plotGraph()

    def plotTriangulation(self):
        scipy.spatial.delaunay_plot_2d(self.tri)

    def plotGraph(self):
        edges = set()
        for v1 in self.edges_dict:
            for v2 in self.edges_dict[v1]:
                if v1 < v2:
                    edges.add( (v1, v2) )
                else:
                    edges.add( (v2, v1) )
        for e in edges:
            plt.plot( (self.circumcenters[e[0],0], self.circumcenters[e[1],0]), (self.circumcenters[e[0],1], self.circumcenters[e[1],1]), 'r')#, 'ro')

        for pt in self.border:
            plt.plot( pt[0], pt[1] , 'bo')

        for s_idx in range(0, self.circumcenters.shape[0], 10):
            cx = self.circumcenters[s_idx,0]
            cy = self.circumcenters[s_idx,1]
            r = self.circumradius[s_idx]
            circle_points_x = []
            circle_points_y = []
            for angle in np.linspace(0.0, math.pi*2.0, 20, endpoint=True):
                circle_points_x.append( math.cos(angle)*r + cx )
                circle_points_y.append( math.sin(angle)*r + cy )
            plt.plot( circle_points_x, circle_points_y)#, 'ro')

        print "edges:", len(edges)
#        plt.show()

    def plotPath(self, path):
        for i in range(len(path)-1):
            print self.circumcenters[path[i],:], self.circumcenters[path[i+1],:]
            plt.plot( (self.circumcenters[path[i],0], self.circumcenters[path[i+1],0]), (self.circumcenters[path[i],1], self.circumcenters[path[i+1],1]), "ro" )
#        plt.show()

    def addEdge(self, e1, e2):
        if not e1 in self.edges_dict:
            self.edges_dict[e1] = set()
        if not e2 in self.edges_dict:
            self.edges_dict[e2] = set()

        self.edges_dict[e1].add( e2 )
        self.edges_dict[e2].add( e1 )

        self.edges_lengths[ (e1,e2) ] = math.sqrt(np.sum(np.square(np.subtract( self.circumcenters[e1,:], self.circumcenters[e2,:] ))))
        self.edges_lengths[ (e2,e1) ] = self.edges_lengths[ (e1,e2) ]

    def getPath(self, v1, v2):
        assert len(self.edges_dict[v1]) != 2
#        if len(self.edges_dict[v2]) != 2:
#            return [v1, v2]
        path = [ v1 ]
        next_node = v2
        current_node = v1
        while len(self.edges_dict[next_node]) == 2:
            path.append( next_node )
            for v in self.edges_dict[next_node]:
                if current_node != v:
                    current_node = next_node
                    next_node = v
                    break
        path.append( next_node )
        return path

    def getPathLength(self, path):
        length = 0.0
        for i in range(len(path)-1):
            length += self.edges_lengths[ (path[i], path[i+1]) ]

    def calculateShortcuts(self):
        shortcut_idx = 0
        # node_2_shortcut_dict maps every node with 2 edges to its corresponding shortcut path
        node_2_shortcut_dict = {}
        shortcut_2_end_nodes = {}
        #edges_dict_tmp = {}
        #for e in self.edges_dict:
        #    edges_dict_tmp[e] = self.edges_dict[e].tolist()

        visited_nodes = set()
        for v1 in self.edges_dict:
            if len(self.edges_dict[v1]) != 2:
                # get all paths from this node
                for v2 in self.edges_dict[v1]:
                    path = self.getPath(v1, v2)
                    #print path

        # create a new graph with shortcuts

    def dijkstra(self, source):
        Q = set()
        #print "len(self.edges_dict)", len(self.edges_dict)
        #print "self.tri.simplices.shape[0]", self.tri.simplices.shape[0]
        #print "dijkstra source", source
        dist = np.full( (self.tri.simplices.shape[0], ), float("inf") )     # Unknown distance from source to v
        prev = np.full( (self.tri.simplices.shape[0], ), -1, dtype=int )    # Previous node in optimal path from source
        for v in self.edges_dict:               # Initialization
            Q.add(v)                            # All nodes initially in Q (unvisited nodes)
        dist[source] = 0.0                      # Distance from source to source
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
            for v in self.edges_dict[u]:
                if not v in Q:                  # where v is still in Q.
                    continue
                alt = dist[u] + self.edges_lengths[ (u,v) ]
                if alt < dist[v]:               # A shorter path to v has been found
                    dist[v] = alt
                    prev[v] = u
        return dist, prev

    def findShortestPath(self, pt_source):
        source_idx = self.tri.find_simplex( np.asarray( [pt_source] ) )[0]
#        target_idx = self.tri.find_simplex( np.asarray( [pt_target] ) )[0]
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
        target_idx = self.tri.find_simplex( np.asarray( [pt_target] ) )[0]
        print "target", target_idx
        self.dist, self.prev = self.dijkstra(target_idx)

    # return list of two indices of points on the edge of two simplices
    def getCommonEdge(self, s1_idx, s2_idx):
        s1 = self.tri.simplices[s1_idx,:]
        s2 = self.tri.simplices[s2_idx,:]
        pt_idx = []
        for v1_idx in range(3):
            for v2_idx in range(3):
                if s1[v1_idx] == s2[v2_idx]:
                    pt_idx.append( s1[v1_idx] )
        assert len(pt_idx) == 2
        return pt_idx

    def getDrivingForce(self, path, robot, target_pos):
        current_pos = robot.robot_body.position

        source_idx = self.tri.find_simplex( np.asarray( [current_pos] ) )[0]

        #print self.tri.find_simplex( np.asarray( [current_pos] ) )
        s_idx = self.tri.find_simplex( np.asarray( [current_pos] ), bruteforce=True )[0]
        n_idx = self.prev[source_idx]
        if n_idx == -1:
                    n = Vec2d(target_pos[0], target_pos[1]) - current_pos
                    n *= 0.1
                    if n.get_length() > 1.0:
                        n = n / n.get_length()
                    return n, 0.0
        else:
#        for i in range(len(path)):
#            if path[i] == s_idx:
#                if i == len(path)-1:
#                    n = Vec2d(target_pos[0], target_pos[1]) - current_pos
#                    n = n / n.get_length()
#                    return n, 0.0

                # get common edge between two simplices
                #s1 = self.tri.simplices[s_idx,:]
                #s2 = self.tri.simplices[n_idx,:]
                #pt_idx = []
                #for v1_idx in range(3):
                #    for v2_idx in range(3):
                #        if s1[v1_idx] == s2[v2_idx]:
                #            pt_idx.append( s1[v1_idx] )
                #assert len(pt_idx) == 2
                pt_idx = self.getCommonEdge(s_idx, n_idx)
                p1 = Vec2d( self.tri.points[pt_idx[0],0], self.tri.points[pt_idx[0],1] )
                p2 = Vec2d( self.tri.points[pt_idx[1],0], self.tri.points[pt_idx[1],1] )
                vec = p1-p2
                #vec = Vec2d( vec[0], vec[1] )

                obs_avoid_force = Vec2d()
                pos_p1_vec = (p1 - current_pos)
                pos_p2_vec = (p2 - current_pos)
                rotate = False
                if pos_p1_vec.get_length()*0.7 < robot.outer_radius:
                    obs_avoid_force = -vec/pos_p1_vec.get_length()
                    rotate = True

                if pos_p2_vec.get_length()*0.7 < robot.outer_radius:
                    obs_avoid_force += vec/pos_p2_vec.get_length()
                    rotate = True

                torque = 0.0
                if rotate:
                    angle1 = vec.get_angle_between( Vec2d(1,0).rotated(robot.robot_body.angle) )
                    angle2 = (-vec).get_angle_between( Vec2d(1,0).rotated(robot.robot_body.angle) )
                    if abs(angle1) < abs(angle2):
                        torque = -angle1
                    else:
                        torque = -angle2
                #print "torque",torque
                #print obs_avoid_force, pos_p1_vec.get_length(), pos_p2_vec.get_length(), robot.outer_radius

                n = Vec2d( vec[1], -vec[0] )
                n = n / n.get_length()
                s1_s2_vec = self.meancenters[n_idx] - self.meancenters[s_idx]
                if n.dot(s1_s2_vec) > 0:
                    return n + obs_avoid_force, torque
                else:
                    return -n + obs_avoid_force, torque
        return None

    def getPushForce(self, push_info, robot, debug_info=None):
        push_loc = self.pushLocations( push_info, robot )

        # get the closest push to the current pose of robot
        # normalize position difference wrt to others
        pos_diff_max = 0.0
        for rob_pos, rob_angle, p_info, rob_cap in push_loc:
            pos_diff = (robot.robot_body.position - rob_pos).get_length()
            if pos_diff > pos_diff_max:
                pos_diff_max = pos_diff

        best_score = 0.0
        best_push = None
        for rob_pos, rob_angle, p_info, rob_cap in push_loc:
            pos_diff = (robot.robot_body.position - rob_pos).get_length()/pos_diff_max
            ang_diff = abs(wrapAnglePI( robot.robot_body.angle - rob_angle )) / math.pi
            contact_ob, force_ob, score_ob = p_info
            score = score_ob - (pos_diff + ang_diff)
            if score > best_score:
                best_score = score
                best_push = (rob_pos, rob_angle, p_info, rob_cap)

        rob_pos, rob_angle, p_info, rob_cap = best_push
        torque = rob_angle - robot.robot_body.angle
        force = rob_pos - robot.robot_body.position + p_info[1]

        robot.debugVisPose(debug_info, rob_pos, rob_angle)

        return force, torque

    def pushLocations(self, push_info, robot):
        # between robot.radius and robot.outer_radius there are some parts of robot

#        robot.outer_radius
#        robot.push_capabilities
#        for i in range(100):
#            contact, force, score = push_info[random.randint(0, len(push_info)-1)]
#            contact += Vec2d(np.random.normal(0.0, 50.0), np.random.normal(0.0, 50.0))

#            for j in range(10):
#                robot.push_capabilities

#        if True:
#                force_ob, contact_ob, score_ob = push_info[0]
#                contact_r, force_r = robot.push_capabilities[0]
                #print "contact_ob", contact_ob, "contact_r", contact_r
        result = []
        for pi in push_info:
            for contact_r, force_r, radius_r in robot.push_capabilities:
                angle = -pi.n.get_angle_between(-force_r)
                pos_rob = pi.pos - contact_r.rotated(angle)
                result.append( (pos_rob, angle, (pi.pos, -pi.n, pi.score), (contact_r, force_r)) )
        return result

    def beginContactForce(self, robot, contact_robot):
        current_pos = robot.robot_body.position

        source_idx = self.tri.find_simplex( np.asarray( [current_pos] ) )[0]

        #print self.tri.find_simplex( np.asarray( [current_pos] ) )
        s_idx = self.tri.find_simplex( np.asarray( [current_pos] ), bruteforce=True )[0]
        n_idx = self.prev[source_idx]
        if n_idx == -1:
                    n = Vec2d(target_pos[0], target_pos[1]) - current_pos
                    n *= 0.1
                    if n.get_length() > 1.0:
                        n = n / n.get_length()
                    return n, 0.0
        else:
#        for i in range(len(path)):
#            if path[i] == s_idx:
#                if i == len(path)-1:
#                    n = Vec2d(target_pos[0], target_pos[1]) - current_pos
#                    n = n / n.get_length()
#                    return n, 0.0

                # get common edge between two simplices
                s1 = self.tri.simplices[s_idx,:]
                s2 = self.tri.simplices[n_idx,:]
                pt_idx = []
                for v1_idx in range(3):
                    for v2_idx in range(3):
                        if s1[v1_idx] == s2[v2_idx]:
                            pt_idx.append( s1[v1_idx] )
                assert len(pt_idx) == 2
                p1 = Vec2d( self.tri.points[pt_idx[0],0], self.tri.points[pt_idx[0],1] )
                p2 = Vec2d( self.tri.points[pt_idx[1],0], self.tri.points[pt_idx[1],1] )
                vec = p1-p2
                #vec = Vec2d( vec[0], vec[1] )

                obs_avoid_force = Vec2d()
                pos_p1_vec = (p1 - current_pos)
                pos_p2_vec = (p2 - current_pos)
                rotate = False
                if pos_p1_vec.get_length()*0.7 < robot.outer_radius:
                    obs_avoid_force = -vec/pos_p1_vec.get_length()
                    rotate = True

                if pos_p2_vec.get_length()*0.7 < robot.outer_radius:
                    obs_avoid_force += vec/pos_p2_vec.get_length()
                    rotate = True

                torque = 0.0
                if rotate:
                    angle1 = vec.get_angle_between( Vec2d(1,0).rotated(robot.robot_body.angle) )
                    angle2 = (-vec).get_angle_between( Vec2d(1,0).rotated(robot.robot_body.angle) )
                    if abs(angle1) < abs(angle2):
                        torque = -angle1
                    else:
                        torque = -angle2
                #print "torque",torque
                #print obs_avoid_force, pos_p1_vec.get_length(), pos_p2_vec.get_length(), robot.outer_radius

                n = Vec2d( vec[1], -vec[0] )
                n = n / n.get_length()
                s1_s2_vec = self.meancenters[n_idx] - self.meancenters[s_idx]
                if n.dot(s1_s2_vec) > 0:
                    return n + obs_avoid_force, torque
                else:
                    return -n + obs_avoid_force, torque
        return None

class Condition:
    def __init__(self):
        self.type = None
        self.name = None
        self.var_name = None
        self.value = None

class ProbabilityGraph:
    def __init__(self):
        self.v = {}
        self.e = {}

    def addEdge(self, v1_id, v2_id, probability):
        if not v1_id in self.e:
            self.e[v1_id] = []
        self.e[v1_id].append( (v2_id, probability) )

class TaskTightPassages:
    def __init__(self, robot, path, space_map):
        self.robot = robot
        tight_passage = False
        min_clearance = float("inf")
        min_edge = None
        passages = []
        for i in range(len(path)-1):
            pt_idx = space_map.getCommonEdge(path[i], path[i+1])
            A = array2Vec2d( space_map.tri.points[pt_idx[0],:])
            B = array2Vec2d( space_map.tri.points[pt_idx[1],:])
            clearance = (A-B).get_length()
            #ori = self.getOrientationPossibilities(A, B, clearance)
            #print path[i], path[i+1], clearance
            if clearance < robot.outer_radius*3.0:
                if not tight_passage:
                    tight_passage = True
                    tight_begin = i
                else:
                    tight_end = i
#                if 
                #TODO
                min_clearance = min(min_clearance, clearance)
                min_edge = (A, B)
            else:
                if tight_passage:
                    tight_passage = False
                    passages.append( (tight_begin, tight_end, min_clearance) )
                    min_clearance = float("inf")
        print "passages", passages

        # for each passage, add a requirement for orientation
        for tight_begin, tight_end, min_clearance in passages:
            if min_clearance < self.robot.outer_radius*2.5:
                # the orientation is most likely restricted here
                cond = Condition()
                cond.type = "requirement"
                cond.name = "tight passage orientation"
                cond.var_name = "robot.orientation"
#            cond.value = 

    def getOrientationPossibilities(self, clearance, A, B):
        # for the robot there are usually two possible orientations with some error margin and
        # possible transition between them with given likehood
        result = ProbabilityGraph()
        if clearance > self.robot.outer_radius*2.5:
            result.v[0] = ("uniform", 0.0, 2.0*math.pi)
        else:
            result.v[0] = ("normal", wrapAnglePI((A-B).get_angle()+math.pi/2.0), math.pi/16.0)
            result.v[1] = ("normal", wrapAnglePI((A-B).get_angle()-math.pi/2.0), math.pi/16.0)
            prob = (clearance - self.robot.outer_radius*2.0) / (self.robot.outer_radius*2.5 - self.robot.outer_radius*2.0)
            prob = min(1.0, max(0.0, prob))
            result.addEdge(0, 1, prob)
            result.addEdge(1, 0, prob)
        return result

def MunkToGame(pos, height):
    return (pos[0], height-pos[1])

def drawDebugInfo(screen, height, debug_info):
    vector_arrow_size = 0.1
    for di in debug_info:
        if di[0] == "vector":
            color = pygame.color.THECOLORS[di[1]]
            A = di[2]
            B = di[3]
            AB = B - A
            BE = Vec2d(AB[1], -AB[0])
            BF = Vec2d(-AB[1], AB[0])
            #BE = E - B
            C = B*(1.0-vector_arrow_size) + A*vector_arrow_size + BE * vector_arrow_size
            #BF = F - B
            D = B*(1.0-vector_arrow_size) + A*vector_arrow_size + BF * vector_arrow_size
            pygame.draw.line(screen, color, MunkToGame( (int(A[0]), int(A[1])), height ), MunkToGame( (int(B[0]), int(B[1])), height ), 1)
            pygame.draw.line(screen, color, MunkToGame( (int(C[0]), int(C[1])), height ), MunkToGame( (int(B[0]), int(B[1])), height ), 1)
            pygame.draw.line(screen, color, MunkToGame( (int(D[0]), int(D[1])), height ), MunkToGame( (int(B[0]), int(B[1])), height ), 1)
        if di[0] == "line":
            color = pygame.color.THECOLORS[di[1]]
            A = di[2]
            B = di[3]
            pygame.draw.line(screen, color, MunkToGame( (int(A[0]), int(A[1])), height ), MunkToGame( (int(B[0]), int(B[1])), height ), 1)
        if di[0] == "circle":
            color = pygame.color.THECOLORS[di[1]]
            C = di[2]
            radius = di[3]
            pygame.draw.circle(screen, color, MunkToGame( (int(C[0]), int(C[1])), height ), int(radius), 1)


