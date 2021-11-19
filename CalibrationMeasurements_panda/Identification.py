#!/usr/bin/python3

from numpy.lib.function_base import disp
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

#Items = calibrationCSVParsing(itemsNames,"Calibration.csv")
Items = calibrationCSVParsing(itemsNames,"GeneratedCalibration.csv")

referencePointsIndices = [int(key[-1]) for key,_ in Items.items() if referencePointsItemsName in key]

N_REFERENCEPOINTS = len(referencePointsIndices)

identificationRatio = 2/3
N_EXPERIMENTS = int(np.floor(len(Items[itemsNames[0]])*identificationRatio))
N_TESTS = len(Items[itemsNames[0]]) - N_EXPERIMENTS

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS

#TBase0 = np.dot(robotBaseTransformation(Items,robotBaseMarkersOffset,robotBasePlaneOffset,robotBaseItemsNames),TBaseN)
TBase0 = np.eye(4)
TBase0Inv = np.linalg.inv(TBase0)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    #TTool0[j] = np.dot(TToolN,TTool0_optiTrack[j])
    TTool0[j] = TTool0_optiTrack[j]

### MOVING MEASUREMENTS TO THE ROBOT BASE FRAME

Pij = np.empty((N_EXPERIMENTS,N_REFERENCEPOINTS,3))
PijTest = np.empty((N_TESTS,N_REFERENCEPOINTS,3))

for referencePointIndex in referencePointsIndices:
    Pij[:,referencePointIndex,:] = np.dot(TBase0Inv,np.vstack((Items[referencePointsItemsName+str(referencePointIndex)][:N_EXPERIMENTS,:3].T,np.zeros((1,N_EXPERIMENTS))))).T[:,:3]
    PijTest[:,referencePointIndex,:] = np.dot(TBase0Inv,np.vstack((Items[referencePointsItemsName+str(referencePointIndex)][N_EXPERIMENTS:,:3].T,np.zeros((1,N_TESTS))))).T[:,:3]

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

def deltaPi(i,q,pi):
    Ti = T(q,pi)
    PRoboti = Ti[:3,-1] 
    deltaPi = Pij[i,:] - PRoboti
    return(deltaPi.flatten())

def step1(pi):
    AT = np.hstack([AijT(Items["jointsAngles"][i],pi) for i in range(N_EXPERIMENTS)])
    A = np.vstack([Aij(Items["jointsAngles"][i],pi) for i in range(N_EXPERIMENTS)])
    P = np.hstack([deltaPi(i,Items["jointsAngles"][i],pi) for i in range(N_EXPERIMENTS)])

    R = np.dot(np.linalg.inv(np.dot(AT,A)),np.dot(AT,P)).flatten()

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
    
    deltaP = np.vstack([np.array([deltaPi(i,Items["jointsAngles"][i],pi0)]).T for i in range(N_EXPERIMENTS)])

    deltapi = np.dot(Jplus,deltaP).T
    print(deltapi)
    return(pi0+deltapi)

### COMPUTE ERROR

def error(TBase,TTool,pi):
    deltapi  = pi - pi0
    return(np.sum(np.sum(np.array([np.array([np.linalg.norm(PijTest[i,j] - T(Items["jointsAngles"][N_EXPERIMENTS+i],pi0,TBase,TTool[j])[:3,-1] - np.dot(Jpi(Items["jointsAngles"][N_EXPERIMENTS+i],pi0,TBase,TTool[j])[:3,:],deltapi.T)) for j in range(N_REFERENCEPOINTS)]) for i in range(N_TESTS)]),axis=0),axis=0))

### MAIN LOOP

pi = pi0
TBase = np.eye(4)
TTool = TTool0
print(error(TBase,TTool,pi))

pi = step2(TBase,TTool)
print(error(TBase,TTool,pi))

"""
for i in range(10):
    #TBase,TTool = step1(pi)
    #print(error(TBase,TTool,pi))

    pi = step2(TBase,TTool)
    print(error(TBase,TTool,pi))
"""

deltapi0 = [0.04341707754820697,0.09459911192543971,0.06425090291770869,0.2944871335579877,0.2645326922696843]



