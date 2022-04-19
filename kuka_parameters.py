#!/usr/bin/python3

import numpy as np

#Number of revolute joints
JOINTS_NUMBER = 6

#Joints axis index (x (1), y (2) or z (3))
JOINTS_AXIS = [3,2,2,1,2,1]

#Angular offset at each joint (yes (1) or no (0))
JOINTS_OFFSET = [0,1,1,1,1,0]

#Links translations axis (along x (1), y (2) or z (3))
LINKS_TRANSLATIONS = [[1,2],
                      [1],
                      [1,3],
                      [2,3],
                      [3],
                      []]

#Links rotations axis (around x (1), y (2) or z (3))
LINKS_ROTATIONS = [[1],
                   [1,3],
                   [3],
                   [3],
                   [3],
                   []]

#Nominal parameters

d1 = 0.750
d2 = 0.350
d3 = 1.250
d4 = 1.100
d5 = 0.055
d6 = 0.230

#deltaq,T,R
pi0 = np.array([d2,0,0,      #1
                0,d3,0,0,   #2
                0,d4,-d5,0,    #3
                0,0,0,0,   #4
                0,0,0])  #6

#Nominal robot base transformation
TBaseN = np.eye(4)
TBaseN[2,-1] = d1

#Nominal robot tool transformation
TToolN = np.array([[0,0,1,d6],
                   [0,-1,0,0],
                   [1,0,0,0],
                   [0,0,0,1]])