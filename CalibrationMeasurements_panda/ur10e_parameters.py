#!/usr/bin/python3

import numpy as np
from optiTrack_parameters import markerRadius

#Number of revolute joints
JOINTS_NUMBER = 6

#Joints axis index (x (1), y (2) or z (3))
JOINTS_AXIS = [3,2,2,2,-3,2]

#Angular offset at each joint (yes (1) or no (0))
JOINTS_OFFSET = [0,1,1,1,1,0]

#Links translations axis (along x (1), y (2) or z (3))
LINKS_TRANSLATIONS = [[1,2],
                      [1],
                      [1],
                      [1,3],
                      [1],
                      []]

#Links rotations axis (around x (1), y (2) or z (3))
LINKS_ROTATIONS = [[1],
                   [1,3],
                   [1,3],
                   [1],
                   [1],
                   []]

#Nominal parameters

d1 = 0.1807
d2 = 0.17415
d3 = 0.6127
d4 = 0.57155 
d5 = 0.11985
d6 = 0.11655

#Per line : deltaq, T, R
pi0 = np.array([0,d2,0,      #1
                0,d3,0,0,   #2
                0,d4,0,0,    #3
                0,0,-d5,0,   #4
                0,0,0])   #5

#Nominal robot base transformation
TBaseN = np.eye(4)
TBaseN[2,-1] = d1

#Nominal robot tool transformation => To match the optitrack frame
TToolN = np.array([[-1/np.sqrt(2),-1/np.sqrt(2),0,0],
                   [0,0,1,d6],
                   [-1/np.sqrt(2),1/np.sqrt(2),0,0],
                   [0,0,0,1]])

itemsNames = ["optiTrackPose0","optiTrackPose1","optiTrackPose2","optiTrackPose3","optiTrackPose4","optiTrackPose5","robotBaseUL","robotBaseUR","robotBaseDL","robotBaseDR","jointsAngles"]

robotBaseMarkersOffset = 0.025 + np.sqrt(markerRadius**2 - (0.008/2)**2)    #TODO
robotBasePlaneOffset = np.array([0,0,0])

robotBaseItemsNames = ["robotBaseUR","robotBaseUL","robotBaseDR","robotBaseDL"]
referencePointsItemsPrefix = "optiTrackPose"