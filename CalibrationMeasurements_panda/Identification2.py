#!/usr/bin/python3

from http.client import LineTooLong
from random import betavariate
from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation
from GeometricModel import *

from importlib import import_module
from sys import argv

from scipy.odr import *
from random import sample,uniform

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

generated = input("Generated measurements ? y/n ")
if(generated == "y"):
    Items = calibrationCSVParsing(itemsNames,"GeneratedMeasurements_" + argv[1] +".csv")
else:
    Items = calibrationCSVParsing(itemsNames,"Calibration_" + argv[1] +".csv")
    

N_DELTAPI = len(pi0)

referencePointsIndices = [int(key[-1]) for key,_ in Items.items() if referencePointsItemsPrefix in key]
N_REFERENCEPOINTS = len(referencePointsIndices)

N_EXPERIMENTS = len(Items[itemsNames[0]])

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS

TRobotBase = robotBaseTransformation(Items,robotBaseMarkersOffset,robotBasePlaneOffset,robotBaseItemsNames)
TBase0 = np.dot(TRobotBase,TBaseN)
TBase0Rot = np.eye(4)
TBase0Rot[:3,:3] = TBase0[:3,:3]
TBase0 = np.dot(np.linalg.inv(TBase0Rot),TBase0)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TToolN,TTool0_marker[j])

### MOVING MEASUREMENTS TO THE ROBOT BASE FRAME

Pij = np.empty((N_EXPERIMENTS,N_REFERENCEPOINTS,3))

for referencePointIndex in referencePointsIndices:
    Pij[:,referencePointIndex,:] = np.dot(np.hstack((Items[referencePointsItemsPrefix+str(referencePointIndex)][:,:3],np.ones((N_EXPERIMENTS,1)))),np.linalg.inv(TBase0Rot))[:,:3]

Jointi = Items["jointsAngles"]

def fromVector(v):
    TBase = np.array([[1,-v[5],v[4],v[0]],
                      [v[5],1,-v[3],v[1]],
                      [-v[4],v[3],1,v[2]],
                      [0,0,0,1]])

    TTool = np.empty((N_REFERENCEPOINTS,4,4))
    for j in range(N_REFERENCEPOINTS):
        TTool[j] = np.eye(4)
        TTool[j][:3,-1] = v[6+j*3:6+j*3+3]

    deltaPi = v[6+3*N_REFERENCEPOINTS:]

    return(TBase,TTool,deltaPi)

def toVector(TBase,TTool,deltaPi):
    v = TBase[:3,-1]
    v = np.append(v,TBase[2,1])
    v = np.append(v,TBase[0,2])
    v = np.append(v,TBase[1,0])

    for j in range(N_REFERENCEPOINTS):
        v = np.append(v,TTool[j][:3,-1])
    
    v = np.append(v,deltaPi)
    return(v)
    
def error(v):
    TBase,TTool,deltaPi = fromVector(v)
    return(np.sum(np.sum(np.array([np.array([np.linalg.norm(Pij[i,j] - T(Jointi[i],pi0+deltaPi,TBase,TTool[j])[:3,-1]) for j in range(N_REFERENCEPOINTS)]) for i in range(N_EXPERIMENTS)]),axis=0),axis=0))

def gradError(v):
    initialVector = deepcopy(v)
    initialError = error(initialVector)
    gradient = np.zeros(len(initialVector))

    for i,_ in enumerate(gradient):
        newVector = v
        newVector[i] += 10e-6
        e = error(newVector)
        print(i)
        print(newVector)
        print(e)
        gradient[i] = (e - initialError)/10e-6
        v = initialVector

    return(gradient)

deltaPi = np.zeros(len(pi0))
TBase = TBase0
TTool = TTool0

def gradDescent(initialVector,learnRate,iterationsNumber):
    v = initialVector
    for i in range(iterationsNumber):
        v -= learnRate*gradError(v)
    return(v)

e = error(toVector(TBase,TTool,deltaPi))
print(e)

finalV = gradDescent(toVector(TBase,TTool,deltaPi),10e-6,5)
e = error(finalV)
print(e)
