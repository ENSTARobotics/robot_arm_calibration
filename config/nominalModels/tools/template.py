#!/usr/bin/python3

import numpy as np

### RENAME FILE : toolName_parameters.py

### FILL PARAMETERS HERE !
#Ordered relative positions of the end effector points according to the holding tool reference frame
#              dX  dY  dZ
d = np.array([[0.0,0.0,0.0],#1
              [1.0,0.0,0.0],#2
              [0.0,-1.0,0.0],#3
              [-1.0,1.0,1.0]])#4

                                         
#  (4)               
#             
#             
#
#            (1)        (2)
#                                     
#                           Y  
#                           |
#            (3)           -|-- X     
#                          (Holding tool reference frame)

### FILL PARAMETERS HERE !
#Homogeneous transformation matrix linking the robot flange to the holding tool reference frame
TToolN_body = np.eye(4)

TToolN = np.array([np.dot(TToolN_body,np.array([[1,0,0,item[0]],
                                    [0,1,0,item[1]],
                                    [0,0,1,item[2]],
                                    [0,0,0,1]])) for item in d])

N_EEPOINTS = len(d)
EEItemsNames = ["EEPositions" + str(i) for i in range(N_EEPOINTS)]

from robot_arm_calibration.ParametersTools import save_parameters
import os
save_parameters(os.path.splitext(os.path.abspath(__file__))[0] + ".yaml", transformation_flange_tool = TToolN)