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

#Distances
d1 = 0.1
d2 = 0.5
d3 = 0.2
d4 = 0.1

#Masses
m1 = 1
m2 = 2
m3 = 3

mBase = 0
mTool = 0

#Center of masses
g1 = np.array([0,0,d2/2])
g2 = np.array([0,d3/2,0])
g3 = np.array([d4/2,0,0])

gBase = np.array([0,0,d1/2])
gTool = np.array([0,0,0])

#Compliances
k1 = np.array([0,0,10])
k2 = np.array([10,0,0])
k3 = np.array([10,0,0])

#deltaq,T,R
pi0 = np.array([0,0,0,  #1
                0,d3,0,0])  #2

#Nominal robot base transformation
TBaseN = np.eye(4)
TBaseN[2,-1] = d1 + d2

#Nominal robot tool transformation
TFlangeN = np.array([[0,0,1,d4],
                     [0,-1,0,0],
                     [1,0,0,0],
                     [0,0,0,1]])