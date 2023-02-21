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
from functools import reduce
import matplotlib.pyplot as plt

plt.rc('font', **{'size': 12, 'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
d = 3.5*(1 + 4/13)
plt.rcParams["figure.figsize"] = [d,(3/4)* d]

### RETRIEVE PARAMETERS FROM DIRECTORY NAME
currentParameters = np.array(re.split('/|_',os.getcwd()))
currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
robotName = currentParameters[0]
markerName = currentParameters[1]

### GET ROBOT CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/nominalModels/" + robotName + "/"
sys.path.append(configPath)

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(robotName + "_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID ROBOT NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

### GET MARKER CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/nominalModels/markers/"
sys.path.append(configPath)

### IMPORT MARKER PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(markerName + "_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID MARKER NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

#Custom packages
from CalibrationCSVParsing import calibrationCSVParsing
from AnalyticGeometricModel import Pfunc,Jfunc

### RETRIEVE MEASUREMENTS DATA

itemsNames = EEPointsItemsNames + ["jointsAngles"]

# Uncalibrated data

testingSamplingMethod = input("Testing sampling method ? " + str(["_".join(folder.split("_")[1:])[:-1] for folder in glob.glob("testing_*/")]))
Items0 = calibrationCSVParsing(itemsNames, "testing_" + testingSamplingMethod + "/measurements/Calibration.csv")

TBase0 = TBaseN

TTool0 = np.empty((N_EEPOINTS,4,4))
for j in range(N_EEPOINTS):
    TTool0[j] = np.dot(TFlangeN,TToolN_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

Jointi0 = Items0["jointsAngles"][:20]
Pij0 = np.empty((len(Jointi0),N_EEPOINTS,3))
for i,item in enumerate(EEPointsItemsNames):
    Pij0[:,i,:] = Items0[item][:20]

def computeGlobalDelta(parameters,Pij=Pij0,Jointi=Jointi0):
    return(Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

def computeGlobalError(parameters,Pij=Pij0,Jointi=Jointi0):
    delta = computeGlobalDelta(parameters,Pij,Jointi)
    positionsNumber = np.shape(Pij)[0]*np.shape(Pij)[1]
    return(np.sqrt(np.dot(delta.T,delta)/positionsNumber))

def computeLocalDelta(parameters,endEffectorPointIndex,configurationIndex,Pij=Pij0,Jointi=Jointi0):
    return(Pij[configurationIndex,endEffectorPointIndex].flatten() - Pfunc(parameters,Jointi[configurationIndex]).flatten()[3*endEffectorPointIndex:3*endEffectorPointIndex+3])

def computeLocalError(parameters,endEffectorPointIndex,configurationIndex,Pij=Pij0,Jointi=Jointi0):
    delta = computeLocalDelta(parameters,endEffectorPointIndex,configurationIndex,Pij,Jointi)
    return(np.sqrt(np.dot(delta.T,delta)))

# Calibrated data
learningSamplingMethod = [folder[:-1] for folder in glob.glob("*/") if folder.split("_")[0] != "testing"]
tmpInput = input("Learning configurations sampling methods ? " + str(learningSamplingMethod))
learningSamplingMethod = tmpInput.split(" ") if tmpInput != "*" else learningSamplingMethod
foldersNames = np.array([["/".join(folder.split("/")[1:])[:-1] for folder in glob.glob(method + "/*/measurements/*/")] for method in learningSamplingMethod])
foldersNames = reduce(np.intersect1d, foldersNames)

learningChoiceMethods = np.unique(np.array([folder.split("/")[0] for folder in foldersNames]))
tmpInput = input("Learning configurations choice methods ? " + str(learningChoiceMethods))
learningChoiceMethods = tmpInput.split(" ") if tmpInput != "*" else learningChoiceMethods
foldersNames = np.array([folder for folder in foldersNames if folder.split("/")[0] in learningChoiceMethods])

identificationMethods = np.unique(np.array([folder.split("/")[-1] for folder in foldersNames]))
tmpInput = input("Parameters identification methods ? " + str(identificationMethods))
identificationMethods = tmpInput.split(" ") if tmpInput != "*" else identificationMethods
foldersNames = np.array([folder for folder in foldersNames if folder.split("/")[-1] in identificationMethods])

foldersNames = np.concatenate([np.array([method+"/"+folder for folder in foldersNames]) for method in learningSamplingMethod])

learningScores = False
tmpInput = input("Compute learning scores values ? y/n")
learningScores = True if tmpInput == "y" else False

### UTILITY SCORE

def a_optimality(CalibrationConfigurations):
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    return(np.trace(np.dot(J.T,J)))

def d_optimality(CalibrationConfigurations):
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    return(np.linalg.det(np.dot(J.T,J)))

def e_optimality(CalibrationConfigurations):
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    _,SVD = np.linalg.svd(np.dot(J.T,J))
    return(SVD[-1])

def o1_optimality(CalibrationConfigurations):
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    measurementsNumber = float(len(CalibrationConfigurations))
    return(np.power(np.linalg.det(np.dot(J.T,J)),1/measurementsNumber)/measurementsNumber)

def o2_optimality(CalibrationConfigurations):
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    _,SVD = np.linalg.svd(np.dot(J.T,J))
    return(SVD[-1]/SVD[0])

def o3_optimality(CalibrationConfigurations):
    return(e_optimality(CalibrationConfigurations))

def o4_optimality(CalibrationConfigurations):
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    _,SVD = np.linalg.svd(np.dot(J.T,J))
    return((SVD[-1]**2)/SVD[0])

def o5_optimality(CalibrationConfigurations):
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    _,SVD = np.linalg.svd(np.dot(J.T,J))
    inverseSVD = 1/SVD
    return(1/np.sum(inverseSVD))

#Interest configurations
#TODO Custom definition !
qStar = np.zeros(JOINTS_NUMBER)

def A_optimality_accuracy(CalibrationConfigurations):
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    JStar = np.vstack([Jfunc(parameters0,qStar) for q in CalibrationConfigurations])
    return(1/np.trace(np.dot(np.dot(JStar,np.linalg.inv(np.dot(J.T,J))),JStar.T)))

utilityScore =  ""
if(learningScores):
    tmpInput = input("Utility score function ? ")
    utilityScore = tmpInput.lower()

fig,ax = plt.subplots()

localError = np.empty((len(Jointi0),N_EEPOINTS))
for i,item in enumerate(EEPointsItemsNames):
    for j,joint in enumerate(Jointi0):
        localError[j,i] = computeLocalError(parameters0,i,j)
avgLocalError = np.average(localError,axis=1)
stdLocalError = np.std(localError,axis=1)
ax.errorbar(np.arange(1,len(Jointi0)+1), avgLocalError, stdLocalError, label="uncalibrated", marker='o',linestyle="none",markersize=4,capsize=4)
print("Testing error raw : " + str(computeGlobalError(parameters0)))

samplingChoiceMethod = ""
for folder in foldersNames:

    #label = folder.split("/")[0] + " " + folder.split("/")[1] + " " + folder.split("/")[3]
    label = folder.split("/")[1]
    with open(folder + "/CalibratedModel.yaml",'r') as file:
        calibratedModel = yaml.load(file, Loader=yaml.FullLoader)

    TBase = np.array(calibratedModel["robotBase"]).reshape((4,4))

    TTool = np.empty((N_EEPOINTS,4,4))
    for j in range(N_EEPOINTS):
        TTool[j] = np.dot(np.array(calibratedModel["robotFlange"]).reshape((4,4)),TToolN_marker[j])

    PBase = TBase[:3,-1]
    RBase = R.from_matrix(TBase[:3,:3])
    PTool = TTool[:,:3,-1]
    pi = np.array(calibratedModel["robotParameters"])[np.array(calibratedModel["robotParameters"]) != 0]

    parameters = np.hstack((PBase,R.as_euler(RBase,"xyz"),pi,PTool.flatten()))

    for i,item in enumerate(EEPointsItemsNames):
        for j,joint in enumerate(Jointi0):
            localError[j,i] = computeLocalError(parameters,i,j)
    avgLocalError = np.average(localError,axis=1)
    stdLocalError = np.std(localError,axis=1)
    ax.errorbar(np.arange(1,len(Jointi0)+1), avgLocalError, stdLocalError, label=label, marker='o',linestyle="none",markersize=4,capsize=4)
    print("Testing error for " + label + " : " + str(computeGlobalError(parameters)))

    if(learningScores):
        if(folder.split("/")[1] != samplingChoiceMethod):
            samplingChoiceMethod = folder.split("/")[1]

            Items = calibrationCSVParsing(itemsNames, folder.split("/")[0] + "/" + folder.split("/")[1] + "/measurements/Calibration.csv")

            Jointi = Items["jointsAngles"]
            Pij = np.empty((len(Jointi),N_EEPOINTS,3))
            for i,item in enumerate(EEPointsItemsNames):
                Pij[:,i,:] = Items[item]

            score = locals()[utilityScore.lower()](Jointi)

            print("Identifiability score for " + folder.split("/")[0] + " " + samplingChoiceMethod + " : " + str(score))
            print("Initial learning error for " + folder.split("/")[0] + " " + samplingChoiceMethod + " : " + str(computeGlobalError(parameters0,Pij,Jointi)))
    
        print("Final learning error for " + label + " : " + str(computeGlobalError(parameters,Pij,Jointi))) 

ax.set_xlabel("Verification configurations")
ax.set_xticks(np.append(1,np.arange(5,len(Jointi0)+1,5)))
ax.set_ylabel("Verification error over the " + str(N_EEPOINTS) + " markers (m)")
ax.set_yscale('log', nonpositive="clip")

ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
ax.grid(which = "major", linestyle = '--')
ax.grid(which = "minor", linestyle = '--')

ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)

plt.savefig("./output.pdf",bbox_inches='tight',pad_inches=0.01)