
from importlib import import_module
from sys import argv
import numpy as np

import yaml
from scipy.spatial.transform import Rotation as R

from GeometricModel import *
from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation,errorsCSVParsing

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(argv[1]+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID ROBOT NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

### IMPORT MARKER PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(argv[2]+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID MARKER NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

### LOAD CALIBRATION PARAMETERS
parameters = np.empty((0,0))
try:
    with open("parameters_" + argv[1] + "_" + argv[2] + ".npy","rb") as f:
        parameters = np.load(f)
except:
    print("NO CALIBRATION PARAMETERS AVAILABLE - RUN CALIBRATION FIRST !")
    exit(1)

### LOAD ROBOT BASE MEASUREMENTS
itemsNames = ["optiTrackPose" + str(i) for i in range(len(TTool0_marker))] + robotBaseItemsNames + ["jointsAngles"]
Items = calibrationCSVParsing(itemsNames,"Calibration_" + argv[1] + "_" + argv[2] + ".csv")
TRobotBase = robotBaseTransformation(Items,robotBaseMarkersOffset,robotBasePlaneOffset)

### FILL ROBOT CONFIGURATION
calibratedModel = {}

#Joints axis
calibratedModel["jointsAxis"] = JOINTS_AXIS

#Robot base
PBase = np.array([parameters[:3]]).T
RBase = R.from_euler("xyz",parameters[3:6]).as_matrix()
TBase = np.dot(np.linalg.inv(TRobotBase),np.vstack((np.hstack((RBase,PBase)),np.array([[0,0,0,1]]))))

#TBase = TBaseN

calibratedModel["robotBase"] = TBase.flatten().tolist()

#Robot geometric parameters
pi = parameters[6:6+len(pi0)]

#pi = pi0

piFull = np.zeros(JOINTS_NUMBER*7)

for i,jointAxis in enumerate(JOINTS_AXIS):
    localParameters = pi[PARAMETERS_INDEX[i]:PARAMETERS_INDEX[i+1]]
    currentIndex = 0

    #Joint offset
    if(JOINTS_OFFSET[i] == 1):
        piFull[i*7] = localParameters[0]
        currentIndex += 1

    #Link translations
    for k,axis in enumerate(LINKS_TRANSLATIONS[i]):
        piFull[i*7+1:i*7+4][np.abs(axis)-1] = localParameters[currentIndex]
        currentIndex += 1

    #Link rotations
    for k,axis in enumerate(LINKS_ROTATIONS[i]):
        piFull[i*7+4:i*7+7][np.abs(axis)-1] = localParameters[currentIndex]
        currentIndex += 1

calibratedModel["robotParameters"] = piFull.tolist()

#Robot flange

TTool = np.empty(np.shape(TTool0_marker))
for i,P in enumerate(np.reshape(parameters[6+len(pi0):],(-1,3))):
    TTool[i] = np.eye(4)
    TTool[i][:3,:3] = np.dot(TToolN[:3,:3],TTool0_marker[i][:3,:3])
    TTool[i][:3,-1] = P - np.dot(TToolN[:3,:3],TTool0_marker[i][:3,-1])
TTool = np.average(TTool,axis=0)

#TTool = np.dot(TToolN,TTool0_body)

calibratedModel["robotFlange"] = TTool.flatten().tolist()

### FILL SETUP CONFIGURATION

calibratedModel["baseReference"] = TRobotBase.flatten().tolist()

calibratedModel["toolReferences"] = {}
for i,tool in enumerate(TTool0_marker):
    calibratedModel["toolReferences"][str(i+1)] = np.dot(np.linalg.inv(TTool0_body),tool).flatten().tolist()

calibratedModel["toolReferencesNumber"] = len(TTool0_marker)

with open("CalibratedModel_" + argv[1] + "_" + argv[2] + ".yaml",'w') as file:
    yaml.dump(calibratedModel, file)