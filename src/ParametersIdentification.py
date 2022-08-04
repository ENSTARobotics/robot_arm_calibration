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
from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation,errorsCSVParsing
from AnalyticGeometricModel import Pfunc,Jfunc,size
from NumericGeometricModel import display

### RETRIEVE MEASUREMENTS DATA
itemsNames = robotBaseItemsNames + markerItemsNames + ["jointsAngles"]

Items = calibrationCSVParsing(itemsNames,"Calibration.csv")
    
TRobotBase = robotBaseTransformation(Items)

identificationRatio = 2/3
N_MEASUREMENTS = len(Items["jointsAngles"])
N_LEARNING = int(np.floor(N_MEASUREMENTS*identificationRatio))
N_TESTS = N_MEASUREMENTS - N_LEARNING

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS
TBase0 = np.dot(TRobotBase,TBaseN)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TToolN,TToolN_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

### CREATING LEARNING AND TESTING SUBSETS

print("Test on first measurement :")
for i,item in enumerate(markerItemsNames):
    print("Reference point " + item)
    print(Items[item][0])
    print(Pfunc(parameters0,Items["jointsAngles"][0])[3*i:3*i+3].T[0])

print("Press any key to continue")
a = input()

PijLear = np.empty((N_LEARNING,N_REFERENCEPOINTS,3))
PijTest = np.empty((N_TESTS,N_REFERENCEPOINTS,3))

randomIndices = np.random.permutation(N_LEARNING+N_TESTS)

indexLear = randomIndices[:N_LEARNING]
for i,item in enumerate(markerItemsNames):
    PijLear[:,i,:] = Items[item][indexLear]
JointiLear = Items["jointsAngles"][indexLear]

if(identificationRatio != 1):
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

def computeErrorTest(parameters):
    return(np.linalg.norm(PijTest.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiTest])))

def computeError(parameters):  
    return(np.linalg.norm(PijLear.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiLear])))

def LLS(parameters,Niterations):

    print("LINEAR LEAST SQUARES")

    Error = computeError(parameters)
    print("Initial error : ")
    print(Error)

    if(identificationRatio != 1):
        ErrorTest = computeErrorTest(parameters)
        print("Initial test error : ")
        print(ErrorTest)

    for i in range(Niterations):

        print("Iteration " + str(i+1) + "/" + str(Niterations))

        J = np.vstack([Jfunc(parameters,q) for q in JointiLear])
        Delta = (PijLear.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiLear]))

        parameters += np.dot(np.linalg.pinv(J),Delta)

        #displayResults(parameters)

        #Intermediate error
        Error = computeError(parameters)
        print("Intermediate error : ")
        print(Error)

        if(identificationRatio != 1):
            ErrorTest = computeErrorTest(parameters)
            print("Intermediate test error : ")
            print(ErrorTest)

    #Final error
    print("Final error : ")
    print(Error)

    if(identificationRatio != 1):
        print("Final test error :")
        print(ErrorTest)

    return(parameters)

def GD(parameters,Niterations):

    print("GRADIENT DESCENT")

    Einit = computeError(parameters)
    print("Initial error : ")
    print(Einit)

    for i in range(Niterations):

        print("Iteration " + str(i+1) + "/" + str(Niterations))

        J = np.vstack([Jfunc(parameters,q) for q in JointiLear])
        Delta = (PijLear.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiLear]))

        gradient = -2*np.dot(J.T,Delta)
        gradient /= np.linalg.norm(gradient)

        JtJ = np.dot(J,J.T)
        alpha = np.dot(np.dot(Delta.T,JtJ),Delta)/(2*np.dot(np.dot(Delta.T,np.dot(JtJ,JtJ)),Delta))
        print("alpha = ")
        print(alpha)
    
        parameters -= alpha*gradient

        #displayResults(parameters)

        #Intermediate error
        Einter = computeError(parameters)
        print("Intermediate error : ")
        print(Einter)

    #Final error
    Efinal = computeError(parameters)
    print(Efinal)

    return(parameters)

def LM(parameters,Niterations):

    print("LEVENBERG-MARQUARDT ALGORITHM")

    Einit = computeError(parameters)
    print("Initial error : ")
    print(Einit)

    for i in range(Niterations):

        print("Iteration " + str(i+1) + "/" + str(Niterations))

        J = np.vstack([Jfunc(parameters,q) for q in JointiLear])
        Delta = (PijLear.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiLear]))

        JtJ = np.dot(J,J.T)
        alpha = np.dot(np.dot(Delta.T,JtJ),Delta)/(2*np.dot(np.dot(Delta.T,np.dot(JtJ,JtJ)),Delta))
        print("alpha = ")
        print(alpha)

        tJJ = np.dot(J.T,J)
        parameters += np.dot(np.linalg.inv(tJJ + alpha*np.diag(np.diag(tJJ))),np.dot(J.T,Delta))

        #displayResults(parameters)

        #Intermediate error
        Einter = computeError(parameters)
        print("Intermediate error : ")
        print(Einter)

    #Final error
    Efinal = computeError(parameters)
    print(Efinal)

    return(parameters)

def BFGS(parameters,Niterations):
    
    print("BFGS ALGORITHM")

    Einit = computeError(parameters)
    print("Initial error : ")
    print(Einit)

    B = np.eye(len(parameters))
    J = np.vstack([Jfunc(parameters,q) for q in JointiLear])
    Delta = (PijLear.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiLear]))

    gradient = -2*np.dot(J.T,Delta)

    for i in range(Niterations):

        print("Iteration " + str(i+1) + "/" + str(Niterations))

        #Find direction
        d = -np.dot(np.linalg.inv(B),gradient)

        #Find learning rate
        alpha = np.dot(np.dot(Delta.T,J),d)/np.dot(np.dot(np.dot(d.T,J.T),J),d)

        #Update parameters
        deltaParameters = alpha*d
        parameters += deltaParameters

        J = np.vstack([Jfunc(parameters,q) for q in JointiLear])
        Delta = (PijLear.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in JointiLear]))

        prevGradient = deepcopy(gradient)
        gradient = -2*np.dot(J.T,Delta)

        deltaGradient = np.array([gradient - prevGradient]).T
        B += np.dot(deltaGradient,deltaGradient.T)/np.dot(deltaGradient.T,deltaParameters) - np.dot(np.dot(np.dot(B,deltaParameters),deltaParameters.T),B)/np.dot(np.dot(deltaParameters.T,B),deltaParameters)

        #displayResults(parameters)

        #Intermediate error
        Einter = computeError(parameters)
        print("Intermediate error : ")
        print(Einter)

    #Final error
    Efinal = computeError(parameters)
    print(Efinal)

    return(parameters)

### MAIN LOOP

print("Number of experiments : " + str(N_LEARNING))
print("Number of tests : " + str(N_TESTS))

#print("Nominal values : ")
#displayResults(parameters0)

#Theorical values

generated = input("Generated measurements ?")
if(generated == "y"):
    parametersTH = errorsCSVParsing('GeneratedErrors.csv')
    print("Theorical values")
    displayResults(parametersTH)
    print("Theorical error")
    print(computeError(parametersTH))
    print("Theorical test error")
    print(computeErrorTest(parametersTH))

Niterations = 5
try : 
    Niterations = int(sys.argv[1])
except:
    print("No iterations number given, defaulting to " + str(Niterations))

optimizationMethod = "LLS"
try : 
    optimizationMethod = sys.argv[2]
except:
    print("No optimization method given, defaulting to " + optimizationMethod)

parameters = locals()[optimizationMethod.upper()](parameters0,Niterations)

### FILL ROBOT CALIBRATED MODEL

print("Saving robot calibrated model...")

calibratedModel = {}

#Joints axis
calibratedModel["jointsAxis"] = JOINTS_AXIS

#Robot base
PBase = np.array([parameters[:3]]).T
RBase = R.from_euler("xyz",parameters[3:6]).as_matrix()
TBase = np.dot(np.linalg.inv(TRobotBase),np.vstack((np.hstack((RBase,PBase)),np.array([[0,0,0,1]]))))

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
    TTool[i] = np.dot(TToolN,TToolN_marker[i])
    TTool[i][:3,-1] = P
    TTool[i] = np.dot(TTool[i],np.linalg.inv(TToolN_marker[i]))
TTool = np.average(TTool,axis=0)

calibratedModel["robotFlange"] = TTool.flatten().tolist()

### FILL SETUP CONFIGURATION

"""
calibratedModel["baseReference"] = TRobotBase.flatten().tolist()

calibratedModel["toolReferences"] = {}
for i,tool in enumerate(TToolN_marker):
    calibratedModel["toolReferences"][str(i+1)] = np.dot(np.linalg.inv(TToolN_body),tool).flatten().tolist()

calibratedModel["toolReferencesNumber"] = len(TToolN_marker)
"""

if(not os.path.isdir(os.getcwd() + "/" + optimizationMethod.upper() + "_" + str(Niterations) + "/")):
        os.mkdir(os.getcwd() + "/" + optimizationMethod.upper() + "_" + str(Niterations) + "/")
with open(os.getcwd() + "/" + optimizationMethod.upper() + "_" + str(Niterations) + "/CalibratedModel.yaml",'w') as file:
    yaml.dump(calibratedModel, file)
print("Done !")

