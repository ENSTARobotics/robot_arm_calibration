#!/usr/bin/python3

import numpy as np

markerDepth = 0.1

#dX,dY
d = np.array([[0.1,-0.1],[-0.1,0.1]])

TToolN_body = np.eye(4)
TToolN = np.array([np.dot(TToolN_body,np.array([[1,0,0,item[0]],
                                    [0,1,0,item[1]],
                                    [0,0,1,markerDepth],
                                    [0,0,0,1]])) for item in d])

N_EEPOINTS = len(d)
EEItemsNames = ["EEPositions" + str(i) for i in range(len(d))]

from robot_arm_calibration.ParametersTools import saveParameters
import os
saveParameters(os.path.splitext(os.path.abspath(__file__))[0] + ".yaml",TTool=TToolN)