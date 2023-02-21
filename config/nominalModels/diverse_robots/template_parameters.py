#!/usr/bin/python3

import numpy as np
import sys
import os

### RENAME FILE : robotName_parameters.py

robotName = os.path.basename(__file__).split("_")[0]

### FILL PARAMETERS HERE !
#Per line : alpha, d, r, theta - Modified Denavit Hartenberg parameters convention !
DHParameters = np.array([[0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0]])

### Import routine
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from DHtoModel import DHtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = DHtoModel(DHParameters,robotName)