#!/usr/bin/python3

import numpy as np
import sys
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

if(not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/" + robotName + "_parameters.npy")):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from DHtoModel import DHtoModel
    DHtoModel(DHParameters,robotName)

with open(os.path.dirname(os.path.realpath(__file__)) + "/" + robotName + "_parameters.npy","rb") as file:
    JOINTS_NUMBER = np.load(file, allow_pickle=True)
    JOINTS_AXIS = np.load(file, allow_pickle=True)
    JOINTS_OFFSET = np.load(file, allow_pickle=True)
    LINKS_TRANSLATIONS = np.load(file, allow_pickle=True)
    LINKS_ROTATIONS = np.load(file, allow_pickle=True)
    pi0 = np.load(file, allow_pickle=True)
    TBaseN = np.load(file, allow_pickle=True)
    TFlangeN = np.load(file, allow_pickle=True)