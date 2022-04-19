#!/usr/bin/python3

import numpy as np

toolDepth = 0.019

#dX,dY,dZ
d = np.array([[-0.015,-0.050,0.00565],
              [-0.075,-0.065,0.00565],
              [-0.05,0.095,0.00565],
              [0.07,0.06,0.00565],
              [0.08,-0.08,0.00565],
              [0.014,0.002,0.01605],
              [-0.002,0.02,0.04005]])

TTool0_marker = np.array([np.array([[1,0,0,item[0]],
                              [0,1,0,item[1]],
                              [0,0,1,toolDepth+item[2]],
                              [0,0,0,1]]) for item in d])
