#!/usr/bin/python3

import numpy as np
import sys
import os

robotName = os.path.basename(__file__).split("_")[0]

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from DHtoModel import DHtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = DHtoModel(DHParameters,robotName)