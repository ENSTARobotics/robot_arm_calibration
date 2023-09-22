#!/usr/bin/python3

#System packages
import yaml
import glob
import os

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
from functools import reduce
import matplotlib.pyplot as plt

### Import robot and marker parameters
from robot_arm_calibration.ParametersTools import importParameters, getIdentificationParameters
importParameters()

#Custom packages
from robot_arm_calibration.CalibrationCSVParsing import calibrationCSVParsing
from robot_arm_calibration.AnalyticGeometricModel import Pfunc

### Retrieve measurements data

itemsNames = EEPointsItemsNames + ["jointsPositions"]

### Uncalibrated data
testingSamplingMethod = input("Testing configurations ? " + str([folder[:-1] for folder in glob.glob("*/")]))
Items0 = calibrationCSVParsing(itemsNames,os.getcwd() + "/" + testingSamplingMethod + "/measurements/Calibration.csv")

parameters0 = getIdentificationParameters(TBaseN,TFlangeN,TToolN_marker,pi0)

Jointi0 = Items0["jointsPositions"][:20]
Pij0 = np.empty((len(Jointi0),N_EEPOINTS,3))
for i,item in enumerate(EEPointsItemsNames):
    Pij0[:,i,:] = Items0[item][:20]

### Tool functions

def computeGlobalDelta(parameters,Pij=Pij0,Jointi=Jointi0):
    """! Computes the global deltas between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The deltas vector.
    """

    return(Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

def computeGlobalError(parameters,Pij=Pij0,Jointi=Jointi0):
    """! Computes the global error (i.e. euclidian distance) between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The computed error.
    """

    delta = computeGlobalDelta(parameters,Pij,Jointi)
    positionsNumber = np.shape(Pij)[0]*np.shape(Pij)[1]
    return(np.sqrt(np.dot(delta.T,delta)/positionsNumber))

def computeAxisSpecificGlobalError(parameters,Pij=Pij0,Jointi=Jointi0):
    delta = computeGlobalDelta(parameters,Pij,Jointi)
    positionsNumber = np.shape(Pij)[0]*np.shape(Pij)[1]

    globalErrorX = np.sqrt(np.dot(delta[::3].T,delta[::3])/positionsNumber)
    globalErrorY = np.sqrt(np.dot(delta[1::3].T,delta[1::3])/positionsNumber)
    globalErrorZ = np.sqrt(np.dot(delta[2::3].T,delta[2::3])/positionsNumber)

    return(globalErrorX,globalErrorY,globalErrorZ)

def computeAxisSpecificGlobalDifference(parameters,Pij=Pij0,Jointi=Jointi0):
    delta = computeGlobalDelta(parameters,Pij,Jointi)

    globalDifferenceX = np.mean(delta[::3])
    globalDifferenceX_std = np.std(delta[::3])
    globalDifferenceY = np.mean(delta[1::3])
    globalDifferenceY_std = np.std(delta[1::3])
    globalDifferenceZ = np.mean(delta[2::3])
    globalDifferenceZ_std = np.std(delta[2::3])

    return(globalDifferenceX,globalDifferenceX_std,globalDifferenceY,globalDifferenceY_std,globalDifferenceZ,globalDifferenceZ_std)

def computeLocalDelta(parameters,endEffectorPointIndex,configurationIndex,Pij=Pij0,Jointi=Jointi0):
    """! Computes the local deltas between the measured and modeled positions.

    @param parameters               The current robot parameters vector.
    @param endEffectorPointIndex    The the end effector point index.
    @param configurationIndex       The configuration index.
    @param Pij                      The measured positions vector.
    @param Jointi                   The measured joints positions vector.

    @return  The deltas vector.
    """

    return(Pij[configurationIndex,endEffectorPointIndex].flatten() - Pfunc(parameters,Jointi[configurationIndex]).flatten()[3*endEffectorPointIndex:3*endEffectorPointIndex+3])

def computeLocalError(parameters,endEffectorPointIndex,configurationIndex,Pij=Pij0,Jointi=Jointi0):
    """! Computes the local error (i.e. euclidian distance) between the measured and modeled positions.

    @param parameters               The current robot parameters vector.
    @param endEffectorPointIndex    The the end effector point index.
    @param configurationIndex       The configuration index.
    @param Pij                      The measured positions vector.
    @param Jointi                   The measured joints positions vector.

    @return  The computed error.
    """

    delta = computeLocalDelta(parameters,endEffectorPointIndex,configurationIndex,Pij,Jointi)
    return(np.sqrt(np.dot(delta.T,delta)))

def computeAxisSpecificLocalError(parameters,endEffectorPointIndex,configurationIndex,Pij=Pij0,Jointi=Jointi0):
    delta = computeLocalDelta(parameters,endEffectorPointIndex,configurationIndex,Pij,Jointi)

    localErrorX = np.abs(delta[0])
    localErrorY = np.abs(delta[1])
    localErrorZ = np.abs(delta[2])

    return(localErrorX,localErrorY,localErrorZ)
    
### Calibrated data

#Sampling method selection
learningSamplingMethod = [folder[:-1] for folder in glob.glob("*/")]
tmpInput = input("Learning configurations sampling methods ? " + str(learningSamplingMethod))
learningSamplingMethod = tmpInput.split(" ") if tmpInput != "*" else learningSamplingMethod
foldersNames = np.array([[folder for folder in glob.glob(method + "/*/measurements/*/")] for method in learningSamplingMethod])
foldersNames = reduce(np.intersect1d, foldersNames)

#Selection method selection
learningSelectionMethods = np.unique(np.array([folder.split("/")[1] for folder in foldersNames]))
tmpInput = input("Learning configurations selection methods ? " + str(learningSelectionMethods))
learningSelectionMethods = tmpInput.split(" ") if tmpInput != "*" else learningSelectionMethods
foldersNames = np.array([folder for folder in foldersNames if folder.split("/")[1] in learningSelectionMethods])

#Identification method selection
identificationMethods = np.unique(np.array([folder.split("/")[3] for folder in foldersNames]))
tmpInput = input("Parameters identification methods ? " + str(identificationMethods))
identificationMethods = tmpInput.split(" ") if tmpInput != "*" else identificationMethods
foldersNames = np.array([folder for folder in foldersNames if folder.split("/")[3] in identificationMethods])

#Sorting folders names...
foldersNames = np.sort(foldersNames)

learningScores = False
tmpInput = input("Compute learning scores values ? y/n")
learningScores = True if tmpInput == "y" else False

### Utility score
utilityScore =  ""
if(learningScores):
    from robot_arm_calibration.TargetConfigurationsGeneration import *
    tmpInput = input("Utility score function ? " + str(utilityScores))
    utilityScore = tmpInput.lower()

fig,ax = plt.subplots()

#Plot uncalibrated error
localError = np.empty((len(Jointi0),N_EEPOINTS))
for i,item in enumerate(EEPointsItemsNames):
    for j,joint in enumerate(Jointi0):
        localError[j,i] = computeLocalError(parameters0,i,j)

avgLocalError = np.average(localError,axis=1)
stdLocalError = np.std(localError,axis=1)
ax.errorbar(np.arange(1,len(Jointi0)+1), avgLocalError, stdLocalError, label="uncalibrated", marker='o',linestyle="none",markersize=4,capsize=4)
print("Testing error uncalibrated : " + str(computeGlobalError(parameters0)) + " min value : " + str(np.min(localError)) + " max value : " + str(np.max(localError)) + " avg value : " + str(np.average(localError.flatten())) + " std value : " + str(np.std(localError.flatten())))

print("Axis specific global error uncalibrated : " + str(computeAxisSpecificGlobalError(parameters0)))
print("Axis specific global difference uncalibrated : " + str(computeAxisSpecificGlobalDifference(parameters0)))

#Plot calibrated errors
samplingSelectionMethod = ""
for folder in foldersNames:
    print("Folder : " + folder)
    label = folder.split("/")[0] + " " + folder.split("/")[1] + " " + folder.split("/")[3]
    with open(folder + "/CalibratedModel.yaml",'r') as file:
        calibratedModel = yaml.load(file, Loader=yaml.FullLoader)

    parameters = getIdentificationParameters(np.array(calibratedModel["robotBase"]).reshape((4,4)), np.array(calibratedModel["robotFlange"]).reshape((4,4)), TToolN_marker,np.array(calibratedModel["robotParameters"])[np.array(calibratedModel["robotParameters"]) != 0])

    for i,item in enumerate(EEPointsItemsNames):
        for j,joint in enumerate(Jointi0):
            localError[j,i] = computeLocalError(parameters,i,j)

    avgLocalError = np.average(localError,axis=1)
    stdLocalError = np.std(localError,axis=1)
    ax.errorbar(np.arange(1,len(Jointi0)+1), avgLocalError, stdLocalError, label=label, marker='o',linestyle="none",markersize=4,capsize=4)
    print("Testing error " + label + " : " + str(computeGlobalError(parameters)) + " min value : " + str(np.min(localError)) + " max value : " + str(np.max(localError)) + " avg value : " + str(np.average(localError.flatten())) + " std value : " + str(np.std(localError.flatten())))

    print("Axis specific global error " + label + " : " + str(computeAxisSpecificGlobalError(parameters)))
    print("Axis specific global difference " + label + " : " + str(computeAxisSpecificGlobalDifference(parameters)))

    #Compute learning scores
    if(learningScores):

        #Only compute initial scores for new selection methods !
        if(folder.split("/")[1] != samplingSelectionMethod):
            samplingSelectionMethod = folder.split("/")[1]

            Items = calibrationCSVParsing(itemsNames, folder.split("/")[0] + "/" + folder.split("/")[1] + "/measurements/Calibration.csv")

            Jointi = Items["jointsPositions"]
            Pij = np.empty((len(Jointi),N_EEPOINTS,3))
            for i,item in enumerate(EEPointsItemsNames):
                Pij[:,i,:] = Items[item]

            score = locals()[utilityScore](Jointi)

            print("Identifiability score for " + folder.split("/")[0] + " " + samplingSelectionMethod + " : " + str(score))
            print("Initial learning error for " + folder.split("/")[0] + " " + samplingSelectionMethod + " : " + str(computeGlobalError(parameters0,Pij,Jointi)))
    
        print("Final learning error for " + label + " : " + str(computeGlobalError(parameters,Pij,Jointi))) 

#Make plot fancy and all
ax.set_xlabel("Verification configurations")
ax.set_xticks(np.append(1,np.arange(5,len(Jointi0)+1,5)))
ax.set_ylabel("Verification error over the " + str(N_EEPOINTS) + " markers (m)")
ax.set_yscale('log', nonpositive="clip")

ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
ax.grid(which = "major", linestyle = '--')
ax.grid(which = "minor", linestyle = '--')

ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)

plt.savefig("./ResultsDisplayOutput.pdf",bbox_inches='tight',pad_inches=0.01)