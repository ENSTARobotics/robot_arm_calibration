#!/usr/bin/python3

import numpy as np

#Number of revolute joints
JOINTS_NUMBER = 2

#Joints axis index (x (1), y (2) or z (3))
JOINTS_AXIS = [3,2]

#Angular offset at each joint (yes (1) or no (0))
JOINTS_OFFSET = [1,1]

#Links translations axis (along x (1), y (2) or z (3))
LINKS_TRANSLATIONS = [[1,2],
                      []]

#Links rotations axis (around x (1), y (2) or z (3))
LINKS_ROTATIONS = [[1],
                   []]

#Nominal parameters

d = 10

#deltaq,T,R
pi0 = np.array([0,d,0,0,  #1
                0])  #2

#Nominal robot base transformation
TBaseN = np.eye(4)

#Nominal robot tool transformation
TFlangeN = np.eye(4)