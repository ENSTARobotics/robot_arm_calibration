#!/usr/bin/python3

#System packages
from importlib import import_module
import sys
import os
from copy import deepcopy
import yaml
import glob

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
import re

### GET CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/"
sys.path.append(configPath)

### RETRIEVE PARAMETERS FROM DIRECTORY NAME
currentParameters = np.array(re.split('/|_',os.getcwd()))
currentParameters = currentParameters[np.where(currentParameters == "calibration")[0][0] + 1:]

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(currentParameters[0] + "_parameters")

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
    module = import_module(currentParameters[1] + "_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID MARKER NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

#Custom packages
from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation
from AnalyticGeometricModel import Pfunc

### RETRIEVE MEASUREMENTS DATA

itemsNames = robotBaseItemsNames + markerItemsNames + ["jointsAngles"]

# Uncalibrated data

testingSamplingMethod = "naive"
try:
    testingSamplingMethod = sys.argv[1]
except:
    print("No testing sampling method provided, defaulting to " + testingSamplingMethod)

solvingMethod = "LLS"
try:
    solvingMethod = sys.argv[2]
except:
    print("No solving method provided, defaulting to " + solvingMethod)

solvingIterations = 5
try:
    solvingIterations = int(sys.argv[3])
except:
    print("Invalid solving iterations number provided, defaulting to " + solvingIterations)

fileName = glob.glob("/".join(os.getcwd().split("/")[:-1]) + "/testing_" + testingSamplingMethod + "_*/measurements/Calibration.csv")[0]
Items0 = calibrationCSVParsing(itemsNames, fileName)

TRobotBase = robotBaseTransformation(Items0)

TBase0 = np.dot(TRobotBase,TBaseN)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TFlangeN,TToolN_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

Jointi = Items0["jointsAngles"]
Pij = np.empty((len(Jointi),N_REFERENCEPOINTS,3))
for i,item in enumerate(markerItemsNames):
    Pij[:,i,:] = Items0[item]

def computeGlobalDelta(parameters):
    return(Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

def computeGlobalError(parameters):  
    delta = computeGlobalDelta(parameters)
    return(np.dot(delta.T,delta))

def computeLocalDelta(parameters,referencePointIndex,configurationIndex):
    return(Pij[configurationIndex,referencePointIndex].flatten() - Pfunc(parameters,Jointi[configurationIndex]).flatten()[3*referencePointIndex:3*referencePointIndex+3])

def computeLocalError(parameters,referencePointIndex,configurationIndex):
    delta = computeLocalDelta(parameters,referencePointIndex,configurationIndex)
    return(np.dot(delta.T,delta))

# Calibrated data

foldersNames = glob.glob(os.getcwd()+"/*/")

Parameters = []
Labels = []

for folder in foldersNames:
    print("Loading robot calibrated model with method " + folder.split("/")[-2] + " ...")

    try:
        with open(folder + "measurements/" + solvingMethod.upper() + "_" + str(solvingIterations) + "/CalibratedModel.yaml",'r') as file:
            calibratedModel = yaml.load(file, Loader=yaml.FullLoader)
        print("Done !")
    except:
        print("Cannot find calibrated model with method " + folder.split("/")[-2] + " - Skipping to next method")
        continue

    TBase = np.dot(TRobotBase,np.array(calibratedModel["robotBase"]).reshape((4,4)))

    TTool = np.empty((N_REFERENCEPOINTS,4,4))
    for j in range(N_REFERENCEPOINTS):
        TTool[j] = np.dot(np.array(calibratedModel["robotFlange"]).reshape((4,4)),TToolN_marker[j])

    PBase = TBase[:3,-1]
    RBase = R.from_matrix(TBase[:3,:3])
    PTool = TTool[:,:3,-1]
    pi = np.array(calibratedModel["robotParameters"])[np.array(calibratedModel["robotParameters"]) != 0]

    Parameters.append(np.hstack((PBase,R.as_euler(RBase,"xyz"),pi,PTool.flatten())))
    Labels.append(folder.split("/")[-2])

### GLOBAL ERROR

#print("Global error : ")
#print(computeGlobalError())
#print(computeGlobalError0())

### LOCAL ERROR
import matplotlib.pyplot as plt
fig,ax = plt.subplots()

for n,parameters in enumerate(Parameters):
    localError = np.empty((len(Jointi),N_REFERENCEPOINTS))
    for i,item in enumerate(markerItemsNames):
        for j,joint in enumerate(Jointi):
            localError[j,i] = computeLocalError(parameters,i,j)
    sumLocalError = np.sum(localError,axis=1)
    ax.plot(np.arange(1,len(Jointi)+1), sumLocalError, label=Labels[n])
    print("Total error " + Labels[n] + " : " + str(computeGlobalError(parameters)))

localError = np.empty((len(Jointi),N_REFERENCEPOINTS))
for i,item in enumerate(markerItemsNames):
    for j,joint in enumerate(Jointi):
        localError[j,i] = computeLocalError(parameters0,i,j)
sumLocalError = np.sum(localError,axis=1)
ax.plot(np.arange(1,len(Jointi)+1), sumLocalError, label="uncalibrated")
print("Total error raw : " + str(computeGlobalError(parameters0)))
plt.xlabel("Measurements")
plt.ylabel("Total error over the markers (m)")
plt.legend()

plt.show()