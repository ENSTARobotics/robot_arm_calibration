#!/usr/bin/python3

from xml.etree.ElementTree import PI
from numpy import vstack
from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation
from GeometricModel import *

from importlib import import_module
from sys import argv

from scipy.odr import *

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

identificationRatio = 4/5
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
#randomIndices = []
#with open("random.npy","rb") as f:
#    randomIndices = np.load(f)

indexExperiments = randomIndices[:N_EXPERIMENTS]
indexTest = randomIndices[-N_TESTS:]

for referencePointIndex in referencePointsIndices:
    Pij[:,referencePointIndex,:] = Items[referencePointsItemsPrefix+str(referencePointIndex)][indexExperiments,:3]
    PijTest[:,referencePointIndex,:] = Items[referencePointsItemsPrefix+str(referencePointIndex)][indexTest,:3]

Jointi = Items["jointsAngles"][indexExperiments]
JointiTest = Items["jointsAngles"][indexTest]

### LINEAR MODEL - Order 0 and order 1 terms

# Usefull matrices

#Robot base
PBase0 = TBase0[:3,-1]
RBase0 = TBase0[:3,:3]

#Tool
PTool0 = TTool0[:,:3,-1]

# Initial solution vector

Vinit = np.zeros(6+3*N_REFERENCEPOINTS+len(pi0))

def M(q,pi):
    TRobot = T(q,pi)
    RRobot = TRobot[:3,:3]
    PRobot = TRobot[:3,-1]

    J = Jpi(q,pi)

    M = np.hstack([np.vstack([np.eye(3)]*N_REFERENCEPOINTS), #deltaPBase
                   np.vstack([-np.dot(RBase0,(skewMatrix(PRobot)+np.dot(RRobot,skewMatrix(PTool)))) for PTool in PTool0]), #phiBase
                   np.kron(np.eye(N_REFERENCEPOINTS,dtype=int),np.dot(RBase0,RRobot)), #deltaPTool
                   np.vstack([np.dot(RBase0,(J[:3,:] - np.dot(skewMatrix(PTool),J[3:,:]))) for PTool in PTool0])])  #deltapi

    return(M)

def V(q,pi):
    TRobot = T(q,pi)
    RRobot = TRobot[:3,:3]
    PRobot = TRobot[:3,-1]

    V = np.hstack([PBase0 + np.dot(RBase0,PRobot) + np.dot(np.dot(RBase0,RRobot),PTool) for PTool in PTool0])

    return(V)

def displayResults(V):
    print("BASE") 
    print("TBase = ")
    TBase = np.dot(TBase0,np.array([[1, -V[5], V[4], V[0]],
                                    [V[5], 1, -V[3], V[1]],
                                    [-V[4], V[3], 1, V[2]],
                                    [0,0,0,1]]))
    print(TBase)
    display(TBase)
    print("TOOLS")
    for i,tool in enumerate(TTool0):
        TTool = np.dot(tool,np.array([[1, 0, 0, V[6+3*i]],
                                      [0, 1, 0, V[6+3*i+1]],
                                      [0, 0, 1, V[6+3*i+2]],
                                      [0,0,0,1]]))
        print("TTool"+str(i+1)+" = ")
        display(TTool)
    print("DELTAPI")
    print("deltapi = ")
    display(V[6+3*N_REFERENCEPOINTS:])

# Initial error
Einit = np.linalg.norm(PijTest.flatten() - np.hstack([V(q,pi0) for q in JointiTest]) - np.dot(np.vstack([M(q,pi0) for q in JointiTest]),Vinit))

Mtot = np.vstack([M(q,pi0) for q in Jointi])
Delta = (Pij.flatten() - np.hstack([V(q,pi0) for q in Jointi])).T

Vsol = np.dot(np.linalg.pinv(Mtot),Delta)

# Final error
Efinal = np.linalg.norm(PijTest.flatten() - np.hstack([V(q,pi0) for q in JointiTest]) - np.dot(np.vstack([M(q,pi0) for q in JointiTest]),Vsol))
