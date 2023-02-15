#!/usr/bin/python3

#System packages
from importlib import import_module
import sys
import os
from copy import deepcopy
import yaml

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import basinhopping
import re
import scipy.odr as odr

import time

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
from AnalyticGeometricModel import Pfunc,Jfunc,size
from NumericGeometricModel import display

### RETRIEVE MEASUREMENTS DATA
itemsNames = markerItemsNames + ["jointsAngles"]

Items = calibrationCSVParsing(itemsNames,"Calibration.csv")

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS
TBase0 = TBaseN

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TFlangeN,TToolN_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

#print("Nominal values : ")
#displayResults(parameters0)

### CREATING LEARNING AND TESTING SUBSETS

N_MEASUREMENTS = len(Items["jointsAngles"])
N_LEARNING = min(N_MEASUREMENTS,2*len(parameters0))
N_TESTS = 0

print("Number of experiments : " + str(N_LEARNING))
print("Number of tests : " + str(N_TESTS))

#print("Test on first measurement :")
#for i,item in enumerate(markerItemsNames):
#    print("Reference point " + item)
#    print(Items[item][0])
#    print(Pfunc(parameters0,Items["jointsAngles"][0])[3*i:3*i+3].T[0])
#
#print("Press any key to continue")
#a = input()

PijLear = np.empty((N_LEARNING,N_REFERENCEPOINTS,3))
PijTest = np.empty((N_TESTS,N_REFERENCEPOINTS,3))

randomIndices = np.random.permutation(N_LEARNING+N_TESTS)
randomIndices = np.arange(N_LEARNING+N_TESTS)

indexLear = randomIndices[:N_LEARNING]
for i,item in enumerate(markerItemsNames):
    PijLear[:,i,:] = Items[item][indexLear]
JointiLear = Items["jointsAngles"][indexLear]

if(N_TESTS != 0):
    indexTest = randomIndices[-N_TESTS:]
    for i,item in enumerate(markerItemsNames):
        PijTest[:,i,:] = Items[item][indexTest]
    JointiTest = Items["jointsAngles"][indexTest]

def displayResults(parameters):
    print("BASE") 
    print("TBase = ")
    PBase = np.array([parameters[:3]]).T
    RBase = R.from_euler("xyz",parameters[3:6]).as_matrix()
    display(np.vstack((np.hstack((RBase,PBase)),np.array([[0,0,0,1]]))))

    print("PI")
    print("pi = ")
    display(parameters[6:6+len(pi0)])

    print("TOOLS")
    for i,P in enumerate(np.reshape(parameters[6+len(pi0):],(-1,3))):
        print("TTool"+str(i+1)+" = ")
        TTool = np.eye(4)
        TTool[:3,-1] = P
        display(TTool)

def computeDeltaTest(parameters):
    return(PijTest.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiTest]))

def computeErrorTest(parameters):
    delta = computeDeltaTest(parameters)
    return(np.dot(delta.T,delta))

def computeDelta(parameters):
    return(PijLear.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiLear]))

def computeError(parameters):  
    delta = computeDelta(parameters)
    return(np.dot(delta.T,delta))

def computeJacobian(parameters):
    return(np.vstack([Jfunc(parameters,q) for q in JointiLear]))

def computeGradient(parameters):
    delta = PijLear.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiLear])
    gradient = -2*np.dot(computeJacobian(parameters).T,delta)
    #gradient /= np.linalg.norm(gradient)
    return(gradient)

def computeAlpha(parameters):
    delta = computeDelta(parameters)
    J = computeJacobian(parameters)
    JtJ = np.dot(J,J.T)
    alpha = np.dot(np.dot(delta.T,JtJ),delta)/(2*np.dot(np.dot(delta.T,np.dot(JtJ,JtJ)),delta))
    return(alpha)

def LLS(parameters,Niterations):

    print("LINEAR LEAST SQUARES")

    error = 0
    errorTest = None

    for i in range(Niterations):

        print("Iteration " + str(i+1) + "/" + str(Niterations))

        parameters += np.dot(np.linalg.pinv(computeJacobian(parameters)),computeDelta(parameters))

        #Intermediate error
        error = computeError(parameters)
        print("Intermediate error :  " + str(error))

        if(N_TESTS != 0):
            errorTest = computeErrorTest(parameters)
            print("Intermediate test error :  " + str(errorTest))

    return(parameters,error,errorTest)

def GD(parameters,Niterations):

    print("GRADIENT DESCENT")

    error = 0
    errorTest = None

    for i in range(Niterations):

        print("Iteration " + str(i+1) + "/" + str(Niterations))

        alpha = computeAlpha(parameters)
        parameters -= alpha*computeGradient(parameters)

        #Intermediate error
        error = computeError(parameters)
        print("Intermediate error :  " + str(error))

        if(N_TESTS != 0):
            errorTest = computeErrorTest(parameters)
            print("Intermediate test error :  " + str(errorTest))

    return(parameters,error,errorTest)

def LM(parameters,Niterations):

    print("LEVENBERG-MARQUARDT ALGORITHM")

    error = 0
    errorTest = None

    for i in range(Niterations):

        print("Iteration " + str(i+1) + "/" + str(Niterations))

        delta = computeDelta(parameters)
        J = computeJacobian(parameters)
        JtJ = np.dot(J,J.T)
        alpha = np.dot(np.dot(delta.T,JtJ),delta)/(2*np.dot(np.dot(delta.T,np.dot(JtJ,JtJ)),delta))

        tJJ = np.dot(J.T,J)
        parameters += np.dot(np.linalg.inv(tJJ + alpha*np.diag(np.diag(tJJ))),np.dot(J.T,delta))

        #Intermediate error
        error = computeError(parameters)
        print("Intermediate error :  " + str(error))

        if(N_TESTS != 0):
            errorTest = computeErrorTest(parameters)
            print("Intermediate test error :  " + str(errorTest))

    return(parameters,error,errorTest)

def BFGS(parameters,Niterations):
    
    print("BFGS ALGORITHM")

    error = 0
    errorTest = None

    B = np.eye(len(parameters))

    delta = computeDelta(parameters)
    J = computeJacobian(parameters)
    gradient = -2*np.dot(J.T,delta)

    for i in range(Niterations):

        print("Iteration " + str(i+1) + "/" + str(Niterations))

        #Find direction
        d = -np.dot(np.linalg.inv(B),gradient)

        #Find learning rate
        alpha = np.dot(np.dot(delta.T,J),d)/np.dot(np.dot(np.dot(d.T,J.T),J),d)

        #Update parameters
        deltaParameters = alpha*d
        parameters += deltaParameters

        delta = computeDelta(parameters)
        J = computeJacobian(parameters)

        prevGradient = deepcopy(gradient)
        gradient = -2*np.dot(J.T,delta)

        deltaGradient = np.array([gradient - prevGradient]).T
        B += np.dot(deltaGradient,deltaGradient.T)/np.dot(deltaGradient.T,deltaParameters) - np.dot(np.dot(np.dot(B,deltaParameters),deltaParameters.T),B)/np.dot(np.dot(deltaParameters.T,B),deltaParameters)

        #Intermediate error
        error = computeError(parameters)
        print("Intermediate error :  " + str(error))

        if(N_TESTS != 0):
            errorTest = computeErrorTest(parameters)
            print("Intermediate test error :  " + str(errorTest))

    return(parameters,error,errorTest)

def BH(parameters,Niterations):
    print("BASIN-HOPPING ALGORITHM WITH BFGS")

    error = 0
    errorTest = None

    def function(parameters):
        return(computeError(parameters),computeGradient(parameters))

    def callback(x,f,accept):
        #Intermediate error
        print("Intermediate error : " + str(f))

        if(N_TESTS != 0):
            errorTest = computeErrorTest(x)
            print("Intermediate test error :  " + str(errorTest))

        return(accept)

    result = basinhopping(function,x0=np.zeros(len(parameters0)),niter=Niterations,stepsize=10,minimizer_kwargs={"method":"L-BFGS-B","jac":True},disp=True,callback=callback)
    parameters = result.x

    error = result.f
    errorTest = computeErrorTest(result.x)

    return(parameters,error,errorTest)

### MAIN LOOP

t0 = time.time()

optimizationMethod = "LLS"
try : 
    optimizationMethod = sys.argv[1]
except:
    print("No optimization method given, defaulting to " + optimizationMethod)

Niterations = 5
try : 
    Niterations = int(sys.argv[2])
except:
    print("No iterations number given, defaulting to " + str(Niterations))

#Initial error
initialError = computeError(parameters0)
print("Initial error : " + str(initialError))

if(N_TESTS != 0):
    errorTest = computeErrorTest(parameters0)
    print("Initial test error : " + str(errorTest))

parameters,finalError,finalTestError = locals()[optimizationMethod.upper()](parameters0,Niterations)

#Final error
print("Final error : " + str(finalError))

if(N_TESTS != 0):
    print("Final test error : " + str(finalTestError))

### FILL ROBOT CALIBRATED MODEL

print("Saving robot calibrated model...")

calibratedModel = {}

#Joints axis
calibratedModel["jointsAxis"] = JOINTS_AXIS.tolist()

#Robot base
PBase = np.array([parameters[:3]]).T
RBase = R.from_euler("xyz",parameters[3:6]).as_matrix()
TBase = np.vstack((np.hstack((RBase,PBase)),np.array([[0,0,0,1]])))

calibratedModel["robotBase"] = TBase.flatten().tolist()

#Robot geometric parameters
pi = parameters[6:6+len(pi0)]

piFull = np.zeros(JOINTS_NUMBER*7)

PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER+1)])

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

TTool = np.empty(np.shape(TToolN_marker))
for i,P in enumerate(np.reshape(parameters[6+len(pi0):],(-1,3))):
    TTool[i] = np.dot(TFlangeN,TToolN_marker[i])
    TTool[i][:3,-1] = P
    TTool[i] = np.dot(TTool[i],np.linalg.inv(TToolN_marker[i]))
TTool = np.average(TTool,axis=0)

calibratedModel["robotFlange"] = TTool.flatten().tolist()

### FILL SETUP CONFIGURATION

"""
calibratedModel["baseReference"] = np.eye(4)

calibratedModel["toolReferences"] = {}
for i,tool in enumerate(TToolN_marker):
    calibratedModel["toolReferences"][str(i+1)] = np.dot(np.linalg.inv(TToolN_body),tool).flatten().tolist()

calibratedModel["toolReferencesNumber"] = len(TToolN_marker)
"""

calibratedModel["initialError"] = initialError
calibratedModel["finalError"] = finalError

if(not os.path.isdir(os.getcwd() + "/" + optimizationMethod.upper() + "_" + str(Niterations) + "/")):
    os.mkdir(os.getcwd() + "/" + optimizationMethod.upper() + "_" + str(Niterations) + "/")
with open(os.getcwd() + "/" + optimizationMethod.upper() + "_" + str(Niterations) + "/CalibratedModel.yaml",'w') as file:
    yaml.dump(calibratedModel, file)
print("Done !")

print("-- Parameters identified in " + str(time.time() - t0) + " seconds --")

