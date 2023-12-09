#!/usr/bin/python3

import numpy as np

#Number of revolute joints
JOINTS_NUMBER = 3

#Joints axis index (x (1), y (2) or z (3))
JOINTS_AXIS = [3,2,2]

#Angular offset at each joint (yes (1) or no (0))
JOINTS_OFFSET = [0,1,0]

#Links translations axis (along x (1), y (2) or z (3))
LINKS_TRANSLATIONS = [[1,2],
                      [1],
                      []]

#Links rotations axis (around x (1), y (2) or z (3))
LINKS_ROTATIONS = [[1],
                   [1,3],
                   []]

#Nominal parameters

d1 = 0.1
d2 = 0.5
d3 = 0.2
d4 = 0.1

#deltaq,T,R
piN = np.array([0,0,0,  #1
                0,d3,0,0])  #2

#Nominal robot base transformation
TBaseN = np.eye(4)
TBaseN[2,-1] = d1 + d2

#Nominal robot tool transformation
TFlangeN = np.array([[0,0,1,d4],
                   [0,-1,0,0],
                   [1,0,0,0],
                   [0,0,0,1]])