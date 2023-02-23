#!/usr/bin/python3

import numpy as np
import os

### RENAME FILE : robotName_parameters.py

### FILL PARAMETERS HERE !
urdfFilePath = "..."

### Import routine
from robot_arm_calibration.URDFtoModel import URDFtoModel
JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN = URDFtoModel(urdfFilePath)