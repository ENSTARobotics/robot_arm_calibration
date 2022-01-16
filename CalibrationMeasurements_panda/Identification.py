#!/usr/bin/python3

from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation
from GeometricModel import *

from importlib import import_module
from sys import argv

from optiTrack_parameters import *

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

### RETRIEVE MEASUREMENTS DATA

generated = input("Generated measurements ? y/n ")
if(generated == "y"):
    Items = calibrationCSVParsing(itemsNames,"GeneratedCalibration_" + argv[1] +".csv")
else:
    Items = calibrationCSVParsing(itemsNames,"Calibration_" + argv[1] +".csv")
    

referencePointsIndices = [int(key[-1]) for key,_ in Items.items() if referencePointsItemsPrefix in key]
N_REFERENCEPOINTS = len(referencePointsIndices)

identificationRatio = 4/5
N_EXPERIMENTS = int(np.floor(len(Items[itemsNames[0]])*identificationRatio))
N_TESTS = len(Items[itemsNames[0]]) - N_EXPERIMENTS

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS

TBase0 = np.dot(robotBaseTransformation(Items,robotBaseMarkersOffset,robotBasePlaneOffset,robotBaseItemsNames),TBaseN)
TBase0Inv = np.linalg.inv(TBase0)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TToolN,TTool0_optiTrack[j])

### MOVING MEASUREMENTS TO THE ROBOT BASE FRAME

Pij = np.empty((N_EXPERIMENTS,N_REFERENCEPOINTS,3))
PijTest = np.empty((N_TESTS,N_REFERENCEPOINTS,3))

randomIndices = np.random.permutation(N_EXPERIMENTS+N_TESTS)

for referencePointIndex in referencePointsIndices:
    Pij[:,referencePointIndex,:] = np.dot(TBase0Inv,np.vstack((Items[referencePointsItemsPrefix+str(referencePointIndex)][randomIndices[:N_EXPERIMENTS],:3].T,np.ones((1,N_EXPERIMENTS))))).T[:,:3]
    PijTest[:,referencePointIndex,:] = np.dot(TBase0Inv,np.vstack((Items[referencePointsItemsPrefix+str(referencePointIndex)][randomIndices[-N_TESTS:],:3].T,np.ones((1,N_TESTS))))).T[:,:3]

### STEP 1 - PBase, RBase & PToolj

def Aij(q,pi):
    Ti = T(q,pi)
    PRoboti = Ti[:3,-1]
    PRobotiTildeT = np.array([[0,PRoboti[2],-PRoboti[1]],
                         [-PRoboti[2],0,PRoboti[0]],
                         [PRoboti[1],-PRoboti[0],0]])
    RRoboti = Ti[:3,:3]

    Aij = np.hstack([np.vstack([np.eye(3)]*N_REFERENCEPOINTS),np.vstack([PRobotiTildeT]*N_REFERENCEPOINTS),np.kron(np.eye(N_REFERENCEPOINTS,dtype=int),RRoboti)])
    return(Aij)

def AijT(q,pi):
    Ti = T(q,pi)
    PRoboti = Ti[:3,-1]
    PRobotiTilde = np.array([[0,-PRoboti[2],PRoboti[1]],
                         [PRoboti[2],0,-PRoboti[0]],
                         [-PRoboti[1],PRoboti[0],0]])
    RRobotiT = Ti[:3,:3].T
    
    AijT = np.vstack([np.hstack([np.eye(3)]*N_REFERENCEPOINTS),np.hstack([PRobotiTilde]*N_REFERENCEPOINTS),np.kron(np.eye(N_REFERENCEPOINTS,dtype=int),RRobotiT)])
    return(AijT)

def step1(pi):
    AT = np.hstack([AijT(Items["jointsAngles"][i],pi) for i in range(N_EXPERIMENTS)])
    A = np.vstack([Aij(Items["jointsAngles"][i],pi) for i in range(N_EXPERIMENTS)])
    deltaPN = np.hstack([(Pij[i,:] - T(Items["jointsAngles"][i],pi)[:3,-1]).flatten() for i in range(N_EXPERIMENTS)])

    R = np.dot(np.linalg.inv(np.dot(AT,A)),np.dot(AT,deltaPN)).flatten()

    TBase = np.eye(4) + np.array([[0,-R[5],R[4],R[0]],
                         [R[5],0,-R[3],R[1]],
                         [-R[4],R[3],0,R[2]],
                         [0,0,0,0]])

    TTool = np.empty((N_REFERENCEPOINTS,4,4))
    for j in range(N_REFERENCEPOINTS):
        TTool[j] = np.eye(4)
        TTool[j][:3,-1] = np.dot(TBase[:3,:3].T,R[6+j*3:9+j*3])

    return(TBase,TTool)

### STEP 2 - pi

def step2(TBase,TTool):
    J = np.vstack([Jpi(Items["jointsAngles"][i],pi0,TBase,TTool[j])[:3,:] for i in range(N_EXPERIMENTS) for j in range(N_REFERENCEPOINTS)])
    Jplus = np.linalg.pinv(J)
    
    deltaP = np.hstack([Pij[i,j] - T(Items["jointsAngles"][i],pi0,TBase,TTool[j])[:3,-1] for i in range(N_EXPERIMENTS) for j in range(N_REFERENCEPOINTS)])
    deltapi = np.dot(Jplus,deltaP).T

    return(pi0+deltapi)

### COMPUTE ERROR

def error(TBase,TTool,pi):
    return(np.sum(np.sum(np.array([np.array([np.linalg.norm(PijTest[-i-1,j] - T(Items["jointsAngles"][-i-1],pi,TBase,TTool[j])[:3,-1]) for j in range(N_REFERENCEPOINTS)]) for i in range(N_TESTS)]),axis=0),axis=0))

### MAIN LOOP

pi = pi0
TBase = np.eye(4)
TTool = TTool0

print("--- INITIAL CONFIGURATION ---")

print("TBase = ")
display(TBase)

for i,M in enumerate(TTool0):
    print("TTool" + str(i) + " = ")
    display(M)

print("pi = " + str(pi))

error(TBase,TTool,pi)
print("Error = " + str(error(TBase,TTool,pi)) + " m")

for i in range(50):
    TBase,TTool = step1(pi)
    pi = step2(TBase,TTool)

    print("--- ITERATION " + str(i+1) +" ---")

    print("TBase = ")
    display(TBase)

    for j,M in enumerate(TTool0):
        print("TTool" + str(j) + " = ")
        display(M)

    print("pi = " + str(pi))

    print("Error = " + str(error(TBase,TTool,pi)) + " m")





