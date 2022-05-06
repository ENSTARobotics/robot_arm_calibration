#!/usr/bin/python3

from numpy import concatenate
from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation
from GeometricModel import *

from importlib import import_module
from sys import argv,exit

import sympy as sp

def skewMatrix(V):
    return(np.array([[0,-V[2],V[1]],[V[2],0,-V[0]],[-V[1],V[0],0]]))

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
    Items = calibrationCSVParsing(itemsNames,"GeneratedMeasurements_" + argv[1] +".csv")
    TRobotBase = np.eye(4)
else:
    itemsNames = ["optiTrackPose" + str(i) for i in range(len(TTool0_marker))] + robotBaseItemsNames + ["jointsAngles"]
    Items = calibrationCSVParsing(itemsNames,"Calibration_" + argv[1] +".csv")
    TRobotBase = robotBaseTransformation(Items,robotBaseMarkersOffset,robotBasePlaneOffset,robotBaseItemsNames)

referencePointsIndices = [int(key[-1]) for key,_ in Items.items() if referencePointsItemsPrefix in key]
N_REFERENCEPOINTS = len(referencePointsIndices)

identificationRatio = 1
N_EXPERIMENTS = int(np.floor(len(Items[itemsNames[0]])*identificationRatio))
N_TESTS = len(Items[itemsNames[0]]) - N_EXPERIMENTS

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS

TBase0 = np.dot(TRobotBase,TBaseN)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TToolN,TTool0_marker[j])

### MOVING MEASUREMENTS TO THE ROBOT BASE FRAME

Pij = np.empty((N_EXPERIMENTS,N_REFERENCEPOINTS,3))
#PijTest = np.empty((N_TESTS,N_REFERENCEPOINTS,3))

randomIndices = np.random.permutation(N_EXPERIMENTS+N_TESTS)
randomIndices = np.arange(0,N_EXPERIMENTS+N_TESTS)
#randomIndices = []
#with open("random.npy","rb") as f:
#    randomIndices = np.load(f)

indexExperiments = randomIndices[:N_EXPERIMENTS]
indexTest = randomIndices[-N_TESTS:]

for referencePointIndex in referencePointsIndices:
    Pij[:,referencePointIndex,:] = Items[referencePointsItemsPrefix+str(referencePointIndex)][indexExperiments,:3]
    #PijTest[:,referencePointIndex,:] = Items[referencePointsItemsPrefix+str(referencePointIndex)][indexTest,:3]

Jointi = Items["jointsAngles"][indexExperiments]
#JointiTest = Items["jointsAngles"][indexTest]

### LINEAR MODEL - Order 0 and order 1 terms

# Usefull matrices/vectors

def M(q,RBase,PTool,pi):
    TRobot = T(q,pi)
    RRobot = TRobot[:3,:3]
    PRobot = TRobot[:3,-1]

    J = Jpi(q,pi)

    M = np.hstack([np.vstack([np.eye(3)]*N_REFERENCEPOINTS), #deltaPBase
                   np.vstack([-np.dot(RBase,(skewMatrix(PRobot)+np.dot(RRobot,skewMatrix(P)))) for P in PTool]), #phiBase
                   np.kron(np.eye(N_REFERENCEPOINTS,dtype=int),np.dot(RBase,RRobot)), #deltaPTool
                   np.vstack([np.dot(RBase,(J[:3,:] - np.dot(skewMatrix(P),J[3:,:]))) for P in PTool])])  #deltapi

    return(M)

def V(q,PBase,RBase,PTool,pi):
    TRobot = T(q,pi)
    RRobot = TRobot[:3,:3]
    PRobot = TRobot[:3,-1]

    V = np.hstack([PBase + np.dot(RBase,PRobot) + np.dot(np.dot(RBase,RRobot),P) for P in PTool])

    return(V)

def displayResults(PBase,RBase,PTool,pi,epsilon):
    print("BASE") 
    print("TBase = ")
    TBase = np.vstack([np.hstack([RBase,np.array([PBase]).T]),np.array([[0,0,0,1]])])
    TBase = np.dot(TBase,np.array([[1, -epsilon[5], epsilon[4], epsilon[0]],
                                    [epsilon[5], 1, -epsilon[3], epsilon[1]],
                                    [-epsilon[4], epsilon[3], 1, epsilon[2]],
                                    [0,0,0,1]]))
    display(TBase)

    print("TOOLS")
    for i,P in enumerate(PTool):
        print("TTool"+str(i+1)+" = ")
        TTool = np.eye(4)
        TTool[:3,-1] = P
        TTool = np.dot(TTool,np.array([[1, 0, 0, epsilon[6+3*i]],
                                      [0, 1, 0, epsilon[6+3*i+1]],
                                      [0, 0, 1, epsilon[6+3*i+2]],
                                      [0,0,0,1]]))

        display(TTool)

    print("PI")
    print("pi = ")
    display(pi + epsilon[6+3*N_REFERENCEPOINTS:])

def computeErrorTest(PBase,RBase,PTool,pi,epsilon):
    return(np.linalg.norm(PijTest.flatten() - np.hstack([V(q,PBase,RBase,PTool,pi) for q in JointiTest]) - np.dot(np.vstack([M(q,RBase,PTool,pi) for q in JointiTest]),epsilon)))

def computeError(PBase,RBase,PTool,pi,epsilon):
    return(np.linalg.norm(Pij.flatten() - np.hstack([V(q,PBase,RBase,PTool,pi) for q in Jointi]) - np.dot(np.vstack([M(q,RBase,PTool,pi) for q in Jointi]),epsilon)))

def updateModel(PBase,RBase,PTool,pi,epsilon):

    #Update robot model
    TBase = np.vstack([np.hstack([RBase,np.array([PBase]).T]),np.array([[0,0,0,1]])])
    TBase = np.dot(TBase,np.array([[1, -epsilon[5], epsilon[4], epsilon[0]],
                                    [epsilon[5], 1, -epsilon[3], epsilon[1]],
                                    [-epsilon[4], epsilon[3], 1, epsilon[2]],
                                    [0,0,0,1]]))
    newPBase = TBase[:3,-1]
    newRBase = TBase[:3,:3]

    newPTool = np.empty(np.shape(PTool))

    for i,P in enumerate(PTool):
        TTool = np.eye(4)
        TTool[:3,-1] = P
        TTool = np.dot(TTool,np.array([[1, 0, 0, epsilon[6+3*i]],
                                    [0, 1, 0, epsilon[6+3*i+1]],
                                    [0, 0, 1, epsilon[6+3*i+2]],
                                    [0,0,0,1]]))
        newPTool[i] = TTool[:3,-1]

    newpi = pi + epsilon[6+3*N_REFERENCEPOINTS:]

    return(newPBase,newRBase,newPTool,newpi)

def LLS(PBaseInit,RBaseInit,PToolInit,piInit,epsilonInit):

    print("LINEAR LEAST SQUARES")

    Einit = computeError(PBaseInit,RBaseInit,PToolInit,piInit,epsilonInit)
    Eprev = Einit
    print("Initial error : ")
    print(Einit)

    Mtot = np.vstack([M(q,RBaseInit,PToolInit,piInit) for q in Jointi])
    Delta = (Pij.flatten() - np.hstack([V(q,PBaseInit,RBaseInit,PToolInit,piInit) for q in Jointi])).T

    epsilon = np.dot(np.linalg.pinv(Mtot),Delta)

    #displayResults(PBase,RBase,PTool,pi,epsilon)

    #Final error
    Efinal = computeError(PBaseInit,RBaseInit,PToolInit,piInit,epsilon)
    print("Final error : ")
    print(Efinal)

    return(epsilon)

def GD(PBaseInit,RBaseInit,PToolInit,piInit,epsilonInit,Niteration):

    print("GRADIENT DESCENT")

    Einit = computeError(PBaseInit,RBaseInit,PToolInit,piInit,epsilonInit)
    Eprev = Einit
    print("Initial error : ")
    print(Einit)

    epsilon = epsilonInit

    Mtot = np.vstack([M(q,RBaseInit,PToolInit,piInit) for q in Jointi])
    Delta = (Pij.flatten() - np.hstack([V(q,PBaseInit,RBaseInit,PToolInit,piInit) for q in Jointi])).T

    for i in range(Niteration):

        epsilonPrev = epsilon

        gradient = 2*np.dot(Mtot.T,np.dot(Mtot,epsilon) - Delta)

        MtM = np.dot(Mtot,Mtot.T)
        MeD = np.dot(Mtot,epsilon) - Delta

        alpha = np.dot(np.dot(MeD.T,MtM),MeD)/(2*np.dot(np.dot(MeD.T,np.dot(MtM,MtM)),MeD))
        print("alpha = ")
        print(alpha)

        epsilon = epsilon - alpha*gradient

        #displayResults(PBase,RBase,PTool,pi,epsilon)

        #Intermediate error
        Einter = computeError(PBaseInit,RBaseInit,PToolInit,piInit,epsilon)
        print("Intermediate error : ")
        print(Einter)

        if(Einter > Eprev):
            print("Optimization method does not converge !")
            return(epsilonPrev)  
        else:
            Eprev = Einter

    #Final error
    Efinal = computeError(PBaseInit,RBaseInit,PToolInit,piInit,epsilon)
    print(Efinal)

    return(epsilon)

def iterativeSplittedLLS(PBaseInit,RBaseInit,PToolInit,piInit,epsilonInit,Niteration):

    print("LINEAR LEAST SQUARES")

    Einit = computeError(PBaseInit,RBaseInit,PToolInit,piInit,epsilonInit)
    print("Initial error : ")
    print(Einit)

    PBase = PBaseInit
    RBase = RBaseInit
    PTool = PToolInit
    pi = piInit
    epsilon = epsilonInit

    for i in range(Niteration):
                
        Mtot = np.vstack([M(q,RBase,PTool,pi) for q in Jointi])
        Delta = (Pij.flatten() - np.hstack([V(q,PBase,RBase,PTool,pi) for q in Jointi])).T

        #Step 1
        Mtot1 = Mtot[:,:6 + 3*N_REFERENCEPOINTS]
        epsilon1 = np.dot(np.linalg.pinv(Mtot1),Delta)
        PBase,RBase,PTool,_ = updateModel(PBase,RBase,PTool,pi,np.concatenate((epsilon1,np.zeros(len(pi)))))

        Mtot = np.vstack([M(q,RBase,PTool,pi) for q in Jointi])
        Delta = (Pij.flatten() - np.hstack([V(q,PBase,RBase,PTool,pi) for q in Jointi])).T

        #Step 2
        Mtot2 = Mtot[:,6 + 3*N_REFERENCEPOINTS:]
        epsilon2 = np.dot(np.linalg.pinv(Mtot2),Delta)
        _,_,_,pi = updateModel(PBase,RBase,PTool,pi,np.concatenate((np.zeros(6 + 3*N_REFERENCEPOINTS),epsilon2)))

        epsilon = np.concatenate((epsilon1,epsilon2))

        #displayResults(PBase,RBase,PTool,pi,epsilon)

        #Intermediate error
        Einter = computeError(PBase,RBase,PTool,pi,epsilon)
        print("Intermediate error : ")
        print(Einter)

    #Final error
    Efinal = computeError(PBase,RBase,PTool,pi,epsilon)
    print("Final error : ")
    print(Efinal)

    return(PBase,RBase,PTool,pi,epsilon)

#Initial values

PBase0 = TBase0[:3,-1]
RBase0 = TBase0[:3,:3]
PTool0 = TTool0[:,:3,-1]
pi0 = pi0
epsilon0 = np.zeros(6 + 3*N_REFERENCEPOINTS + len(pi0))

#iterativeSplittedLLS(PBase0,RBase0,PTool0,pi0,epsilon0,10)

epsilon = GD(PBase0,RBase0,PTool0,pi0,epsilon0,10)
PBase,RBase,PTool,pi = updateModel(PBase0,RBase0,PTool0,pi0,epsilon)

for i in range(10):
    epsilon = GD(PBase,RBase,PTool,pi,epsilon0,10)
    PBase,RBase,PTool,pi = updateModel(PBase,RBase,PTool,pi,epsilon)
