#!/usr/bin/python3

import numpy as np

TToolN = np.array([np.eye(4)])

N_EEPOINTS = 1
EEItemsNames = ["EEPose","AimedPosition"]

from robot_arm_calibration.ParametersTools import saveParameters
import os
saveParameters(os.path.splitext(os.path.abspath(__file__))[0] + ".yaml",TTool=TToolN)