#!/usr/bin/python3

import numpy as np

### RENAME FILE : markerName_parameters.py

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

### FILL PARAMETERS HERE !
#Homogeneous transformation matrix linking the robot flange to the holding tool reference frame
TToolN_body = np.eye(4)

TToolN_marker = np.array([np.dot(TToolN_body,np.array([[1,0,0,item[0]],
                                    [0,1,0,item[1]],
                                    [0,0,1,item[2]],
                                    [0,0,0,1]])) for item in d])

N_EEPOINTS = len(d)
EEPointsItemsNames = ["EEPositions" + str(i) for i in range(N_EEPOINTS)]