#!/usr/bin/python3

import numpy as np

#Number of revolute joints
JOINTS_NUMBER = 7

#Joints axis index (x (1), y (2) or z (3))
JOINTS_AXIS = [3,2,3,-2,3,-2,-3]

#Angular offset at each joint (yes (1) or no (0))
JOINTS_OFFSET = [0,1,1,1,1,1,0]

#Links translations axis (along x (1), y (2) or z (3))
LINKS_TRANSLATIONS = [[1,2],
                      [1,3],
                      [1,2],
                      [1,3],
                      [1,2],
                      [1],
                      []]

#Links rotations axis (around x (1), y (2) or z (3))
LINKS_ROTATIONS = [[3],
                   [1],
                   [1],
                   [3],
                   [3],
                   [3],
                   []]

#Nominal parameters

d1 = 0.333
d2 = 0.316
d3 = 0.0825
d4 = 0.384
d5 = 0.088
d6 = 0.107

#Per line : deltaq, T, R
pi0 = np.array([0,0,0,      #1
                0,0,d2,0,   #2
                0,d3,0,0,    #3
                0,-d3,d4,0,   #4
                0,0,0,0,    #5
                0,d5,0])  #6

#Nominal robot base transformation
TBaseN = np.eye(4)
TBaseN[2,-1] = d1

#Nominal robot tool transformation
TToolN = np.array([[1,0,0,0],
                   [0,-1,0,0],
                   [0,0,1,-d6],
                   [0,0,0,1]])

#TODO REVIEW
robotBasePlaneOffset = 0.00565
robotBaseCenterOffset = np.array([0.011+0.127/2,0,0])
robotBaseItemsNames = ["robotBase1","robotBase2","robotBase3","robotBase4","robotBase5","robotBase6","robotBase7"]