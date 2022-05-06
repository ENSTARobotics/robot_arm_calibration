#!/usr/bin/python3

from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation,errorsCSVParsing
from GeometricModel import *

from importlib import import_module
from sys import argv

from scipy.spatial.transform import Rotation as R

from AnalyticGeometricModel import Pfunc,Jfunc

from copy import deepcopy

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

### RETRIEVE MEASUREMENTS DATA

referencePointsItemsPrefix = "optiTrackPose"

generated = input("Generated measurements ? y/n ")
if(generated == "y"):
    itemsNames = ["optiTrackPose" + str(i) for i in range(len(TTool0_marker))] + ["jointsAngles"]
    Items = calibrationCSVParsing(itemsNames,"GeneratedMeasurements_" + argv[1] + "_" + argv[2] + ".csv")
    TRobotBase = np.eye(4)
else:
    itemsNames = ["optiTrackPose" + str(i) for i in range(len(TTool0_marker))] + robotBaseItemsNames + ["jointsAngles"]
    Items = calibrationCSVParsing(itemsNames,"Calibration_" + argv[1] + "_" + argv[2] + ".csv")
    TRobotBase = robotBaseTransformation(Items,robotBaseMarkersOffset,robotBasePlaneOffset)

referencePointsIndices = [int(key[-1]) for key,_ in Items.items() if referencePointsItemsPrefix in key]
N_REFERENCEPOINTS = len(referencePointsIndices)

identificationRatio = 3/4
N_EXPERIMENTS = int(np.floor(len(Items[itemsNames[0]])*identificationRatio))
N_TESTS = len(Items[itemsNames[0]]) - N_EXPERIMENTS

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS

TBase0 = np.dot(TRobotBase,TBaseN)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TToolN,TTool0_marker[j])

### MOVING MEASUREMENTS TO THE ROBOT BASE FRAME

Pij = np.empty((N_EXPERIMENTS,N_REFERENCEPOINTS,3))
PijTest = np.empty((N_TESTS,N_REFERENCEPOINTS,3))

randomIndices = np.random.permutation(N_EXPERIMENTS+N_TESTS)
#randomIndices = np.arange(0,N_EXPERIMENTS+N_TESTS)
#randomIndices = []
#with open("random.npy","rb") as f:
#    randomIndices = np.load(f)

indexExperiments = randomIndices[:N_EXPERIMENTS]
indexTest = randomIndices[-N_TESTS:]

for referencePointIndex in referencePointsIndices:
    Pij[:,referencePointIndex,:] = Items[referencePointsItemsPrefix+str(referencePointIndex)][indexExperiments]
    PijTest[:,referencePointIndex,:] = Items[referencePointsItemsPrefix+str(referencePointIndex)][indexTest]

Jointi = Items["jointsAngles"][indexExperiments]
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
    return(np.linalg.norm(Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi])))

def LLS(parameters,Niterations):

    print("LINEAR LEAST SQUARES")

    Error = computeError(parameters)
    print("Initial error : ")
    print(Error)

    ErrorTest = computeErrorTest(parameters)
    print("Initial test error : ")
    print(ErrorTest)

    for i in range(Niterations):

        J = np.vstack([Jfunc(parameters,q) for q in Jointi])
        Delta = (Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

        parameters += np.dot(np.linalg.pinv(J),Delta)

        displayResults(parameters)

        #Intermediate error
        Error = computeError(parameters)
        print("Intermediate error : ")
        print(Error)

        ErrorTest = computeErrorTest(parameters)
        print("Intermediate test error : ")
        print(ErrorTest)

    #Final error
    print("Final error : ")
    print(Error)
    print("Final test error :")
    print(ErrorTest)

    return(parameters)

def GD(parameters,Niterations):

    print("GRADIENT DESCENT")

    Einit = computeError(parameters)
    print("Initial error : ")
    print(Einit)

    for i in range(Niterations):

        J = np.vstack([Jfunc(parameters,q) for q in Jointi])
        Delta = (Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

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

        J = np.vstack([Jfunc(parameters,q) for q in Jointi])
        Delta = (Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

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
    J = np.vstack([Jfunc(parameters,q) for q in Jointi])
    Delta = (Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

    gradient = -2*np.dot(J.T,Delta)

    for i in range(Niterations):

        #Find direction
        d = -np.dot(np.linalg.inv(B),gradient)

        #Find learning rate
        alpha = np.dot(np.dot(Delta.T,J),d)/np.dot(np.dot(np.dot(d.T,J.T),J),d)

        #Update parameters
        deltaParameters = alpha*d
        parameters += deltaParameters

        J = np.vstack([Jfunc(parameters,q) for q in Jointi])
        Delta = (Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

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

#Initial values

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

print("Number of experiments : " + str(N_EXPERIMENTS))
print("Number of tests : " + str(N_TESTS))

print("Nominal values : ")
displayResults(parameters0)

#Theorical values

if(generated == "y"):
    parametersTH = errorsCSVParsing('GeneratedErrors_' + argv[1] + '_' + argv[2] + '.csv')
    print("Theorical values")
    displayResults(parametersTH)
    print("Theorical error")
    print(computeError(parametersTH))
    print("Theorical test error")
    print(computeErrorTest(parametersTH))

parameters = LLS(parameters0,5)

with open("parameters_" + argv[1] + "_" + argv[2] + ".npy","wb") as f:
    np.save(f,parameters)

