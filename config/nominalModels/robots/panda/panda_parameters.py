#!/usr/bin/python3

import numpy as np

d1 = 0.333
d2 = 0.316
d3 = 0.0825
d4 = 0.384
d5 = 0.088
d6 = 0.107

#Per line : alpha, d, r, theta - Modified parameters convention !
DHParameters = np.array([[0,0,d1,0],
                         [-np.pi/2,0,0,0],
                         [np.pi/2,0,d2,0],
                         [np.pi/2,d3,0,0],
                         [-np.pi/2,-d3,d4,0],
                         [np.pi/2,0,0,0],
                         [np.pi/2,d5,d6,0]])

### Import routine
from robot_arm_calibration.DHtoModel import DHtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,piN,TBaseN,TFlangeN = DHtoModel(DHParameters)

from robot_arm_calibration.ParametersTools import saveParameters
import os
saveParameters(os.path.splitext(os.path.abspath(__file__))[0] + ".yaml",TBase=TBaseN,TFlange=TFlangeN,pi=piN)