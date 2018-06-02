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

def array2Vec2d(array):
    return Vec2d(array[0], array[1])

def normalizeImage(img):
    v_min = np.min(img)
    v_max = np.max(img)
    result = np.subtract(img, v_min)
    result = np.multiply( result, 255.0/(v_max-v_min) )
    return result

def buildDistanceMap(space, range_x, steps_x, range_y, steps_y):
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

def recursiveGetFreeSpaceBorder(dist_map, visited_map, start_point, border):
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
        recursiveGetFreeSpaceBorder( dist_map, visited_map, pt, border )

def applyDamping( damped_objects_list ):
    for do in damped_objects_list:
        body, lin_damping, rot_damping = do
        body.force -= lin_damping * body.velocity
        body.torque -= rot_damping * body.angular_velocity

def getFreeSpaceBorder(dist_map, start_point):
    border = []
    visited_map = np.zeros( dist_map.shape, dtype=int)

    recursiveGetFreeSpaceBorder(dist_map, visited_map, start_point, border)

    return border

# calculate m-sphere circumcenter
# m is the number of simplex vertices, and total dimension of space is the same
# as number of coordinates in each vertex.
# Returns circumcenter.
def calculateCircumcenter(points):
    # create Cayley-Menger matrix
    cm = np.zeros( (points.shape[0]+1,points.shape[0]+1,) )
    for i in range(points.shape[0]):
        for j in range(i+1, points.shape[0]):
            cm[i+1,j+1] = np.sum(np.square(points[i,:]-points[j,:]))
            cm[j+1,i+1] = cm[i+1,j+1]

    for i in range(1, cm.shape[0]):
        cm[i,0] = 1.0
        cm[0,i] = 1.0

    try:
        cmi = np.linalg.inv(cm)
    except:
        return points[0,:]
    cc = cmi[0,1:]
    cc = np.multiply(cc, 1.0/np.sum(cc))
    point_c = np.zeros( (points.shape[1],) )
    for v_idx in range(points.shape[0]):
        point_c = point_c + points[v_idx,:] * cc[v_idx]

    # verify: distance from all vertices to circumcenter should be the same
    radii = np.zeros( (points.shape[0],) )
    for v_idx in range(points.shape[0]):
        radii[v_idx] = math.sqrt( np.sum( np.square(points[v_idx,:]-point_c) ) )

    return point_c

def wrapAnglePI(angle):
    while angle > math.pi:
        angle = angle - 2.0*math.pi
    while angle < -math.pi:
        angle = angle + 2.0*math.pi
    return angle

def wrapAngle2PI(angle):
    while angle > 2.0*math.pi:
        angle = angle - 2.0*math.pi
    while angle < 0.0:
        angle = angle + 2.0*math.pi
    return angle

def MunkToGame(pos, height):
    return (pos[0], height-pos[1])

def drawDebugCircle(debug_info, color, radius, pos):
    debug_info.append( ("circle", color, pos, radius) )

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

def createSpatialModelForBody(body):
    raise Exception("not implemented")

    x_min = float("inf")
    x_max = float("-inf")
    y_min = float("inf")
    y_max = float("-inf")
    points = []
    for s in body.shapes:
        if type(s) == pymunk.shapes.Segment:
            print "Segment"
            print s.a, s.b, s.radius
            x_min = min(x_min, s.a[0] - s.radius)
            x_min = min(x_min, s.b[0] - s.radius)
            x_max = max(x_max, s.a[0] + s.radius)
            x_max = max(x_max, s.b[0] + s.radius)
            y_min = min(y_min, s.a[1] - s.radius)
            y_min = min(y_min, s.b[1] - s.radius)
            y_max = max(y_max, s.a[1] + s.radius)
            y_max = max(y_max, s.b[1] + s.radius)
            for t in np.linspace(0,1):
                points.append( s.a*t + s.b*(1.0-t) )
        elif type(s) == pymunk.shapes.Circle:
            print "Circle"
            print s.offset, s.radius
            x_min = min(x_min, s.offset[0] - s.radius)
            x_max = max(x_max, s.offset[0] + s.radius)
            y_min = min(y_min, s.offset[1] - s.radius)
            y_max = max(y_max, s.offset[1] + s.radius)

    print "x range", x_min, x_max
    print "y range", y_min, y_max

