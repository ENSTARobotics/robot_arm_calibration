#!/usr/bin/python3

import numpy as np

d1 = 0.1
d2 = 0.5
d3 = 0.2

#Per line : alpha, d, r, theta - Modified parameters convention !
DHParameters = np.array([[0,0,d1 + d2,0],
                         [-np.pi/2,0,0,0],
                         [np.pi/2,0,d3,0]])

### Import routine
from robot_arm_calibration.DHtoModel import DHtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,piN,TBaseN,TFlangeN = DHtoModel(DHParameters)

from robot_arm_calibration.ParametersTools import save_parameters
import os
save_parameters(os.path.splitext(os.path.abspath(__file__))[0] + ".yaml",transformation_base_first_joint = TBaseN, transformation_last_joint_flange = TFlangeN, pi = piN)