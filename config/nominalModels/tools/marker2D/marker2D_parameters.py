#!/usr/bin/python3

import numpy as np

markerDepth = 0.0112

#dX,dY
d = np.array([[0.0265,-0.0615],
              [-0.0412,-0.0629],
              [0.0449,-0.0159],
              [0.0602,0.0159],
              [0.0494,0.0562],
              [-0.0412,0.0612]])

TToolN_body = np.array([[1/np.sqrt(2),1/np.sqrt(2),0,0],
                   [-1/np.sqrt(2),1/np.sqrt(2),0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
TToolN = np.array([np.dot(TToolN_body,np.array([[1,0,0,item[0]],
                                    [0,1,0,item[1]],
                                    [0,0,1,markerDepth],
                                    [0,0,0,1]])) for item in d])

N_EEPOINTS = len(d)
EEItemsNames = ["EEPositions" + str(i) for i in range(len(d))]

from robot_arm_calibration.ParametersTools import save_parameters
import os
save_parameters(os.path.splitext(os.path.abspath(__file__))[0] + ".yaml",transformation_flange_tool = TToolN)