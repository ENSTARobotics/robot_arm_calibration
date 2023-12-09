#!/usr/bin/python3

import numpy as np

markerDepth = 0.019

#dX,dY,dZ
d = np.array([[0.08,-0.08,0.00565],#1
              [0.07,0.06,0.00565],#2
              [-0.05,0.095,0.00565],#3
              [-0.075,-0.065,0.00565],#4
              [-0.015,-0.050,0.00565],#5             
              [0.014,0.002,0.01605],#6
              [-0.002,0.02,0.04005]])#7

#                              (1) (Green tape marker)
#                        
#     (2)                
#              ________ 
#             |     (6)|
#             |(7) -   |    
#              --------    (5)             X
#                                          |
#                                      Y --|- 
#  (3)                         (4)     (Holding tool reference frame)


TToolN_body = np.array([[1/np.sqrt(2),1/np.sqrt(2),0,0],
                   [-1/np.sqrt(2),1/np.sqrt(2),0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
TToolN_marker = np.array([np.dot(TToolN_body,np.array([[1,0,0,item[0]],
                                    [0,1,0,item[1]],
                                    [0,0,1,markerDepth+item[2]],
                                    [0,0,0,1]])) for item in d])

N_EEPOINTS = len(d)
EEPointsItemsNames = ["EEPositions" + str(i) for i in range(N_EEPOINTS)]

from robot_arm_calibration.ParametersTools import saveParameters
import os
saveParameters(os.path.splitext(os.path.abspath(__file__))[0] + "_parameters.yaml",TTool=TToolN_marker)