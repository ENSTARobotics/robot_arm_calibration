#!/usr/bin/python3

import numpy as np
import os

### RENAME FILE : robotName_parameters.py

### FILL PARAMETERS HERE !
#Per line : alpha, d, r, theta - Modified Denavit Hartenberg parameters convention !
DHParameters = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0]])

### Import routine
from robot_arm_calibration.DHtoModel import DHtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = DHtoModel(DHParameters)