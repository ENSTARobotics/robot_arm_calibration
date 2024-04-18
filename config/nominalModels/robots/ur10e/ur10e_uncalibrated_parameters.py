#!/usr/bin/python3

import numpy as np

d1 = 0.1807
d2 = 0.17415
d3 = 0.6127
d4 = 0.57155 
d5 = 0.11985
d6 = 0.11655

#Per line : alpha, d, r, theta - Modified parameters convention !
DHParameters = np.array([[0,0,d1,0],
                         [-np.pi/2,0,d2,0],
                         [0,d3,0,0],
                         [0,d4,0,0],
                         [-np.pi/2,0,d5,0],
                         [np.pi/2,0,d6,np.pi]])

### Import routine
from robot_arm_calibration.DHtoModel import DHtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,piN,TBaseN,TFlangeN = DHtoModel(DHParameters)

from robot_arm_calibration.ParametersTools import save_parameters
import os
save_parameters(os.path.splitext(os.path.abspath(__file__))[0] + ".yaml",transformation_base_first_joint = TBaseN, transformation_last_joint_flange = TFlangeN, pi = piN)