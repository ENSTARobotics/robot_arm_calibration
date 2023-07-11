#!/usr/bin/python3

import numpy as np
import os

robotName = os.path.basename(__file__).split("_")[0]

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

from DHtoModel import DHtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = DHtoModel(DHParameters,robotName)