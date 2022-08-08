#!/usr/bin/python3

import numpy as np
import os

robotName = __file__.split("_")[0]

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

if(not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/" + robotName + "_parameters.npy")):
    from DHtoParameters import DHtoParameters
    DHtoParameters(DHParameters,robotName)

with open(os.path.dirname(os.path.realpath(__file__)) + "/" + robotName + "_parameters.npy","rb") as file:
    JOINTS_NUMBER = np.load(file, allow_pickle=True)
    JOINTS_AXIS = np.load(file, allow_pickle=True)
    JOINTS_OFFSET = np.load(file, allow_pickle=True)
    LINKS_TRANSLATIONS = np.load(file, allow_pickle=True)
    LINKS_ROTATIONS = np.load(file, allow_pickle=True)
    pi0 = np.load(file, allow_pickle=True)
    TBaseN = np.load(file, allow_pickle=True)
    TFlangeN = np.load(file, allow_pickle=True)

#TODO REVIEW
robotBasePlaneOffset = 0.00565
robotBaseCenterOffset = np.array([0.011+0.127/2,0,0])
robotBaseItemsNames = ["robotBase1","robotBase2","robotBase3","robotBase4","robotBase5","robotBase6","robotBase7"]