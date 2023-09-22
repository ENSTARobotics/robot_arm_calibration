#!/usr/bin/python3

#System packages
import yaml
import glob
import os
import copy as copy
import sys

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
from functools import reduce
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("tab20")
markers = ["^","s","o","v","D","x","*","+"]
figsize = (10,9)

plt.rc('font', **{'size': 24, 'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

### Import robot and marker parameters
from robot_arm_calibration.ParametersTools import getIdentifiableParametersPartial

#Custom packages
from robot_arm_calibration.CalibrationCSVParsing import calibrationCSVParsing
from robot_arm_calibration.AnalyticGeometricModel import Pfunc, Tfunc, N_EEPOINTS, TBaseN, TFlangeN, TToolN, piN, EEItemsNames

from robot_arm_calibration.Preconditionning import (
    simplePreconditionning,
    simpleJacobianPreconditionning,
    simpleEulerPreconditionning,
    simpleEulerJacobianPreconditionning,
    sphericPreconditionning,
    sphericJacobianPreconditionning,
)
preconditionningType = "spheric"
defaultPreconditionning = sphericPreconditionning
defaultJacobianPreconditionning = sphericJacobianPreconditionning

import robot_arm_calibration.TargetConfigurationsGeneration

from robot_arm_calibration.ParametersIdentification import computeError

### Retrieve measurements data

itemsNames = EEItemsNames + ["jointsPositions","jointsTorques","EndEffectorWrench"]

### Uncalibrated data
testingSamplingMethod = input("Testing configurations ? " + str([folder[:-1] for folder in glob.glob("*/")]))
Items0 = calibrationCSVParsing(itemsNames,os.getcwd() + "/" + testingSamplingMethod + "/measurements/Calibration.csv")

Jointi0 = Items0["jointsPositions"][1:]
N_MEASUREMENTS = len(Jointi0)
if len(TToolN) > 1:
    Pij0 = np.empty((N_MEASUREMENTS, N_EEPOINTS, 3))
    for i, item in enumerate(EEItemsNames):
        Pij0[:, i, :] = Items0[item][1:]
else:
    Pij0 = Items0[EEItemsNames[0]][1:]

### Set preconditionning
if len(TToolN) > 1:
    preconditionning = None
    jacobianPreconditionning = None
    additionnalParameters = None
else:
    preconditionning = defaultPreconditionning
    jacobianPreconditionning = defaultJacobianPreconditionning
    additionnalParameters = None

try:
    additionnalParameters = Items0[sys.argv[1]]
except:
    print("No additionnal parameters provided...")

estimateTBaseOffset = 0
try:
    estimateTBaseOffset = int(sys.argv[2])
except:
    print("Skipping robot base transform estimation...")

### Robot base, robot parameters and end effector points transformations estimations
if len(TToolN) > 1:
    parameters0 = getIdentifiableParametersPartial(TBaseN, TFlangeN, TToolN, piN)
else:
    parameters0 = getIdentifiableParametersFull(TBaseN, TFlangeN, TToolN, piN)

### Filter outliers (i.e. obviously wrong measurements)
print("Number of experiments before outlier removal : " + str(N_MEASUREMENTS))

Outliers = []
if len(TToolN) > 1:     #TODO Full-pose measurements
    for i,(pij,jointValues) in enumerate(zip(Pij0,Jointi0)):
        error = computeError(parameters0, pij, [jointValues])
        if(error > 0.05):
            print("Outlier detected at index " + str(i) + " : " + str(error))
            Outliers.append(i)
Pij0 = np.delete(Pij0,Outliers,axis=0)
Jointi0 = np.delete(Jointi0,Outliers,axis=0)

N_MEASUREMENTS = len(Jointi0)
print("Number of experiments after outlier removal : " + str(N_MEASUREMENTS))

### Estimate the transformation between the sensor reference and the robot base
if estimateTBaseOffset != 0:
    print("Estimating sensor to base transformation...")
    TBaseSensor = np.eye(4)

    if len(TToolN) > 1:
        EEPoints0 = []
        EEPoints1 = []
        for i, configuration in enumerate(Jointi0):
            TBaseFlange = np.dot(
                TBaseN, np.dot(Tfunc(parameters0, configuration), TFlangeN)
            )
            EEPoints0.extend(
                np.vstack(
                    [
                        np.dot(
                            np.dot(TBaseFlange, TTool), np.array([[0, 0, 0, 1]]).T
                        )[:-1].T
                        for TTool in TToolN
                    ]
                )
            )

        EEPoints0 = np.array(EEPoints0)
        EEPoints1 = np.reshape(Pij0,(-1,3))

        # Does not enforce rotation matrix constraints
        # TSensorBase = np.dot(np.hstack([EEPoints1,np.ones((len(EEPoints1),1))]).T,np.linalg.pinv(np.hstack([np.dot(np.dot(TFlangeBase,TTool),np.array([[0,0,0,1]]).T) for TTool in TToolN])))

        # Enforces rotation matrix constraints !
        EEPoints1Centroid = np.mean(EEPoints1, axis=0)
        EEPoints0Centroid = np.mean(EEPoints0, axis=0)

        U, S, V = np.linalg.svd(
            np.dot(
                (EEPoints0 - EEPoints0Centroid).T, (EEPoints1 - EEPoints1Centroid)
            )
        )

        TSensorBase = np.eye(4)
        TSensorBase[:3, :3] = np.dot(
            np.dot(
                V.T,
                np.diag(
                    np.append(np.ones(len(S) - 1), np.linalg.det(np.dot(V.T, U.T)))
                ),
            ),
            U.T,
        )
        TSensorBase[:3, -1] = EEPoints1Centroid - np.dot(
            TSensorBase[:3, :3], EEPoints0Centroid
        )
        TBaseSensor = np.linalg.inv(TSensorBase)

    else:
        PBaseSensor = []
        RBaseSensor = []
        for i, configuration in enumerate(Jointi0):
            EEPose0 = np.dot(
                TBaseN, np.dot(Tfunc(parameters0, configuration), TFlangeN)
            )
            EEPose0 = np.dot(EEPose0, TToolN[0])

            EEPose1 = np.eye(4)
            EEPose1[:3, -1] = Pij0[:3]
            EEPose1[:3, :3] = R.from_mrp(Pij0[3:]).as_matrix()

            # TBaseSensor = TBaseTool * TToolSensor
            DeltaEEPose = np.dot(EEPose0, np.linalg.inv(EEPose1))
            PBaseSensor.append(DeltaEEPose[:3, -1])
            RBaseSensor.append(DeltaEEPose[:3, :3])

        PBaseSensor = np.array(PBaseSensor)
        RBaseSensor = np.array(RBaseSensor)

        # Compute average transformation over all measurements
        PBaseSensor = np.average(PBaseSensor, axis=0)
        RBaseSensor = np.average(RBaseSensor, axis=0)
        U, _, V = np.linalg.svd(RBaseSensor)
        RBaseSensor = np.dot(U, V)

        TBaseSensor[:3, -1] = PBaseSensor
        TBaseSensor[:3, :3] = RBaseSensor

    print("Sensor to base transformation estimation : \n" + str(TBaseSensor))
    print("Position error : " + str(np.linalg.norm(TBaseSensor[:3, -1])))
    print("Rotation error : " + str(np.arccos((np.trace(TBaseSensor[:3, :3]) - 1) / 2)))

    # Transfering measurements in the estimated robot base frame !
    with open("CalibrationCorrected.csv", "wb+") as file:
        for i, jointsPosition in enumerate(Jointi0):
            if len(TToolN) > 1:
                for j, item in enumerate(EEItemsNames):
                    Pij0[i,j] = np.dot(
                        TBaseSensor, np.append(Pij0[i,j], 1).T
                    )[:3]
                    np.savetxt(
                        file,
                        Pij0[i,j].reshape(1, -1),
                        delimiter=",",
                        fmt="%s",
                    )
            else:
                T = np.eye(4)
                T[:3, -1] = Pij0[i][:3]
                T[:3, :3] = R.from_mrp(Pij0[i][3:]).as_matrix()
                correctedPose = np.dot(TBaseSensor, T)
                Pij0[i] = np.append(correctedPose[:3, -1], R.from_matrix(correctedPose[:3, :3]).as_mrp())
                np.savetxt(
                    file,
                    Pij0[i].reshape(1, -1),
                    delimiter=",",
                    fmt="%s",
                )
            np.savetxt(
                file,
                jointsPosition.reshape(1, -1),
                delimiter=",",
                fmt="%s",
            )
            file.write(str.encode("---\n"))

### Tool functions

def computeGlobalDelta(parameters,Pij=Pij0,Jointi=Jointi0):
    """! Computes the global deltas between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The deltas vector.
    """

    return(Pij.flatten() - np.hstack([np.array(Pfunc(parameters,q)).flatten() for q in Jointi]))

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

    return(globalDifferenceX,globalDifferenceY,globalDifferenceZ,globalDifferenceX_std,globalDifferenceY_std,globalDifferenceZ_std)

def computeLocalDelta(parameters,endEffectorPointIndex,configurationIndex,Pij=Pij0,Jointi=Jointi0):
    """! Computes the local deltas between the measured and modeled positions.

    @param parameters               The current robot parameters vector.
    @param endEffectorPointIndex    The the end effector point index.
    @param configurationIndex       The configuration index.
    @param Pij                      The measured positions vector.
    @param Jointi                   The measured joints positions vector.

    @return  The deltas vector.
    """

    return(Pij[configurationIndex,endEffectorPointIndex].flatten() - np.array(Pfunc(parameters,Jointi[configurationIndex])).flatten()[3*endEffectorPointIndex:3*endEffectorPointIndex+3])

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
    return(np.sqrt(np.dot(delta.T,delta)))  #Norm also works !

def computeAxisSpecificLocalError(parameters,endEffectorPointIndex,configurationIndex,Pij=Pij0,Jointi=Jointi0):
    delta = computeLocalDelta(parameters,endEffectorPointIndex,configurationIndex,Pij,Jointi)

    localErrorX = np.abs(delta[0])
    localErrorY = np.abs(delta[1])
    localErrorZ = np.abs(delta[2])

    return(localErrorX,localErrorY,localErrorZ)
    
### Calibrated data

def plot_data(Items0, **kwargs):

    keepLabels = [1,1,1]

    if(not "learningSamplingMethod" in kwargs):
        #Sampling method selection
        learningSamplingMethod = [folder[:-1] for folder in glob.glob("*/")]
        tmpInput = input("Learning configurations sampling methods ? " + str(learningSamplingMethod))
        learningSamplingMethod = tmpInput.split(" ") if tmpInput != "*" else learningSamplingMethod
    else:
        learningSamplingMethod = kwargs["learningSamplingMethod"]

    foldersNames = []
    for method in learningSamplingMethod:
        foldersNames.extend(glob.glob(method + "/*/measurements/*/")) 
    keepLabels[0] = 1 if len(learningSamplingMethod) > 1 else 0

    if(not "learningSelectionMethods" in kwargs):
        #Selection method selection
        learningSelectionMethods = np.unique(np.array([folder.split("/")[1] for folder in foldersNames]))
        tmpInput = input("Learning configurations selection methods ? " + str(learningSelectionMethods))
        learningSelectionMethods = tmpInput.split(" ") if tmpInput != "*" else learningSelectionMethods
    else:
        learningSelectionMethods = kwargs["learningSelectionMethods"]

    foldersNames = np.array([folder for folder in foldersNames if folder.split("/")[1] in learningSelectionMethods])
    keepLabels[1] = 1 if len(learningSelectionMethods) > 1 else 0

    if(not "identificationMethods" in kwargs):
        #Identification method selection
        identificationMethods = np.unique(np.array([folder.split("/")[3] for folder in foldersNames]))
        tmpInput = input("Parameters identification methods ? " + str(identificationMethods))
        identificationMethods = tmpInput.split(" ") if tmpInput != "*" else identificationMethods
    else:
        identificationMethods = kwargs["identificationMethods"]
    foldersNames = np.array([folder for folder in foldersNames if folder.split("/")[3] in identificationMethods])
    keepLabels[2] = 1 if len(identificationMethods) > 1 else 0

    #Sorting folders names...
    foldersNames = np.sort(foldersNames)

    fig,ax = plt.subplots(figsize=figsize)
    fig_hist,ax_hist = plt.subplots(3,1,figsize=(10,12))
    fig_hist_markers, ax_hist_markers = plt.subplots(4,2,figsize=(20,16))

    #Plot uncalibrated error
    localError = np.empty((len(Jointi0),N_EEPOINTS))
    for i,item in enumerate(EEItemsNames):
        for j,joint in enumerate(Jointi0):
            localError[j,i] = computeLocalError(parameters0,i,j)

    avgLocalError = np.average(localError,axis=1)
    stdLocalError = np.std(localError,axis=1)
    ax.errorbar(np.arange(1,len(Jointi0)+1), avgLocalError, stdLocalError, label="Nominal", marker=markers[0],linestyle="none",markersize=9,capsize=9,linewidth=2,capthick=2)
    #print("Testing error uncalibrated : " + str(computeGlobalError(parameters0)) + " min value : " + str(np.min(localError)) + " max value : " + str(np.max(localError)) + " avg value : " + str(np.average(localError.flatten())) + " std value : " + str(np.std(localError.flatten())))
    uncalibratedError = computeGlobalError(parameters0)
    print("Uncalibrated error : " + str(uncalibratedError))

    from scipy.stats import norm
    localDelta = np.empty((len(Jointi0),N_EEPOINTS,3))
    for i,item in enumerate(EEItemsNames):
        for j,joint in enumerate(Jointi0):
            localDelta[j,i,:] = computeLocalDelta(parameters0,i,j)

    bins = np.linspace(np.min(localDelta),np.max(localDelta),31)
    for i,(ax_local,data) in enumerate(zip(ax_hist,localDelta.T)):
        plot_data = data.flatten()
        (mu, sigma) = norm.fit(plot_data)
        _,bins,_ = ax_local.hist(plot_data,bins=bins,color=cmap2(2*i+1))
        best_fit_line = (bins[1]-bins[0])*len(plot_data)*norm.pdf(bins, mu, sigma)
        ax_local.plot(bins, best_fit_line,color=cmap2(2*i+1),zorder=0,linestyle="--",label="uncalibrated")

    #print("Axis specific global error uncalibrated : " + str(computeAxisSpecificGlobalError(parameters0)))
    #print("Axis specific global difference uncalibrated : " + str(computeAxisSpecificGlobalDifference(parameters0)))
    #print("Axis specific deltas (avg,std) " + "uncalibrated" + " : " + str(np.average(localDelta,axis=(0,1))) + " " + str(np.std(localDelta,axis=(0,1))))

    localDeltaNorm = np.linalg.norm(localDelta,axis=2)
    bins = np.linspace(np.min(localDeltaNorm),np.max(localDeltaNorm),21)
    for i,(ax_local,data) in enumerate(zip(ax_hist_markers.flatten(),localDeltaNorm.T)):
        plot_data = data.flatten()
        _,bins,_ = ax_local.hist(plot_data,bins=bins,color=cmap2(2*i+1),label="uncalibrated")

    #print("Position deltas (avg,std) " + "uncalibrated" + " : " + str(np.average(localDeltaNorm)) + " " + str(np.std(localDeltaNorm)))
    #print("Marker specific position deltas (avg,std) " + "uncalibrated" + " : " + str(np.average(localDeltaNorm,axis=0)) + " " + str(np.std(localDeltaNorm,axis=0)))

    #Plot calibrated errors
    defaultItems0 = copy.deepcopy(Items0)   
    for k,folder in enumerate(foldersNames):

        #Resetting measurements
        Items0 = copy.deepcopy(defaultItems0)

        print("Folder : " + folder)
        label = keepLabels[0]*(folder.split("/")[0] + " ") + keepLabels[1]*(folder.split("/")[1] + " ") + keepLabels[2]*(folder.split("/")[3] + " ")
        label = label[:-1]
        print("Label : " + label)
        if label == "greedy":
            label = "DETMAX"
        elif label == "naive_5":
            label = "joint-wise"
        elif label == "naive_10_scan":
            label = "scan-oriented"

        label = label[0].upper() + label[1:]
        with open(folder + "/CalibratedModel.yaml",'r') as file:
            calibratedModel = yaml.load(file, Loader=yaml.FullLoader)

        parameters = getIdentifiableParametersPartial(np.array(calibratedModel["robotBase"]).reshape((4,4)), np.array(calibratedModel["robotFlange"]).reshape((4,4)), TToolN,np.array(calibratedModel["robotParameters"])[np.array(calibratedModel["robotParameters"]) != 0])

        for i,item in enumerate(EEItemsNames):
            for j,joint in enumerate(Jointi0):
                localError[j,i] = computeLocalError(parameters,i,j)

        #print("Testing error " + label + " : " + str(computeGlobalError(parameters)) + " min value : " + str(np.min(localError)) + " max value : " + str(np.max(localError)) + " avg value : " + str(np.average(localError.flatten())) + " std value : " + str(np.std(localError.flatten())))
        calibratedError = computeGlobalError(parameters)
        print("Calibrated error " + label + " : " + str(calibratedError))
        print("Accuracy improvement : " + str(1-calibratedError/uncalibratedError))
        label += " (" + str(np.round(100*(1-calibratedError/uncalibratedError),1)) + "\%)"

        print("Learning error improvement : " + str(1-calibratedModel["finalError"]/calibratedModel["initialError"]))

        avgLocalError = np.average(localError,axis=1)
        stdLocalError = np.std(localError,axis=1)
        ax.errorbar(np.arange(1,len(Jointi0)+1), avgLocalError, np.minimum(stdLocalError,10**-3), label=label, marker=markers[1+k],linestyle="none",markersize=9,capsize=9,linewidth=2,capthick=2)

        localDelta = np.empty((len(Jointi0),N_EEPOINTS,3))
        for i,item in enumerate(EEItemsNames):
            for j,joint in enumerate(Jointi0):
                localDelta[j,i,:] = computeLocalDelta(parameters,i,j)

        bins = np.linspace(np.min(localDelta),np.max(localDelta),31)
        for i,(ax_local,data) in enumerate(zip(ax_hist,localDelta.T)):
            plot_data = data.flatten()
            (mu, sigma) = norm.fit(plot_data)
            _,bins,_ = ax_local.hist(plot_data,bins=bins,color=cmap2(2*i))
            best_fit_line = (bins[1]-bins[0])*len(plot_data)*norm.pdf(bins, mu, sigma)
            ax_local.plot(bins, best_fit_line,color=cmap2(2*i),label=label)

        #print("Axis specific global error " + label + " : " + str(computeAxisSpecificGlobalError(parameters)))
        #print("Axis specific global difference " + label + " : " + str(computeAxisSpecificGlobalDifference(parameters)))
        #print("Axis specific deltas (avg,std) " + label + " : " + str(np.average(localDelta,axis=(0,1))) + " " + str(np.std(localDelta,axis=(0,1))))

        localDeltaNorm = np.linalg.norm(localDelta,axis=2)
        bins = np.linspace(np.min(localDeltaNorm),np.max(localDeltaNorm),21)
        for i,(ax_local,data) in enumerate(zip(ax_hist_markers.flatten(),localDeltaNorm.T)):
            plot_data = data.flatten()
            _,bins,_ = ax_local.hist(plot_data,bins=bins,color=cmap2(2*i),label=label)

        #print("Position deltas (avg,std) " + label + " : " + str(np.average(localDeltaNorm)) + " " + str(np.std(localDeltaNorm)))
        #print("Marker specific position deltas (avg,std) " + label + " : " + str(np.average(localDeltaNorm,axis=0)) + " " + str(np.std(localDeltaNorm,axis=0)))

    #Make plot fancy and all
    ax.set_xlabel("Verification configurations",labelpad=10)
    ax.set_xticks(np.append(1,np.arange(5,len(Jointi0)+1,5)))
    ax.set_ylabel("Verification error over the " + str(N_EEPOINTS) + " markers (m)", labelpad=10)
    ax.set_yscale('log', nonpositive="clip")

    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.grid(which = "major", linestyle = '--')
    ax.grid(which = "minor", linestyle = '--')

    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", borderaxespad=0, ncol=2,handletextpad=0.1)
    fig.savefig("./" + testingSamplingMethod + "_ResultsDisplayOutput.pdf",dpi = 300, bbox_inches = 'tight')

    from matplotlib.ticker import FormatStrFormatter

    labels = ["x","y","z"]
    for ax_local,label in zip(ax_hist,labels):
        ax_local.set_xlabel(r"$\Delta " + label + " (m)$",labelpad=5)
        ax_local.set_ylabel("Counts",labelpad=10)
        ax_local.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax_local.set_axisbelow(True)
        ax_local.grid(which = "major", linestyle = '--')
        ax_local.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="center", borderaxespad=0, ncol=2)
    fig_hist.subplots_adjust(hspace=0.75)
    fig_hist.savefig("./" + testingSamplingMethod + "_ResultsDisplayOutputHist.pdf",dpi = 300, bbox_inches = 'tight')

    for i,ax_local in enumerate(ax_hist_markers.flatten()):
        ax_local.set_xlabel(r"$\|\Delta P_{}\| (m)$".format(i+1),labelpad=10)
        ax_local.set_ylabel("Counts",labelpad=10)
        ax_local.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax_local.tick_params(axis='x', which='major', pad=10)
        ax_local.set_axisbelow(True)
        ax_local.grid(which = "major", linestyle = '--')
        ax_local.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="center", borderaxespad=0, ncol=2)
    fig_hist_markers.delaxes(ax_hist_markers[-1][-1])
    fig_hist_markers.subplots_adjust(hspace=0.9,wspace=0.12)
    fig_hist_markers.savefig("./" + testingSamplingMethod + "_ResultsDisplayOutputHistMarkers.pdf",dpi = 300, bbox_inches = 'tight')

if __name__ == '__main__':
    kwargs = {}
    kwargs["learningSamplingMethod"] = ["naive_10_scan","naive_5"]
    #kwargs["learningSamplingMethod"] = ["naive_5"]
    kwargs["learningSelectionMethods"] = ["annealing"]
    #kwargs["learningSelectionMethods"] = ["random","greedy","annealing","genetic"]
    kwargs["identificationMethods"] = ["GN"]

    plot_data(Items0,**kwargs)