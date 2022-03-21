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

identificationRatio = 4/5
N_EXPERIMENTS = int(np.floor(len(Items[itemsNames[0]])*identificationRatio))
N_TESTS = len(Items[itemsNames[0]]) - N_EXPERIMENTS

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
PijTest = np.empty((N_TESTS,N_REFERENCEPOINTS,3))

randomIndices = np.random.permutation(N_EXPERIMENTS+N_TESTS)
#randomIndices = []
#with open("random.npy","rb") as f:
#    randomIndices = np.load(f)

indexExperiments = randomIndices[:N_EXPERIMENTS]
indexTest = randomIndices[-N_TESTS:]

for referencePointIndex in referencePointsIndices:
    #Pij[:,referencePointIndex,:] = np.dot(TBase0Inv,np.vstack((Items[referencePointsItemsPrefix+str(referencePointIndex)][indexExperiments,:3].T,np.ones((1,N_EXPERIMENTS))))).T[:,:3]
    #PijTest[:,referencePointIndex,:] = np.dot(TBase0Inv,np.vstack((Items[referencePointsItemsPrefix+str(referencePointIndex)][indexTest,:3].T,np.ones((1,N_TESTS))))).T[:,:3]
    #Pij[:,referencePointIndex,:] = np.dot(Items[referencePointsItemsPrefix+str(referencePointIndex)][indexExperiments,:3],RBase0)
    #PijTest[:,referencePointIndex,:] = np.dot(Items[referencePointsItemsPrefix+str(referencePointIndex)][indexTest,:3],RBase0)
    Pij[:,referencePointIndex,:] = np.dot(np.hstack((Items[referencePointsItemsPrefix+str(referencePointIndex)][indexExperiments,:3],np.ones((len(indexExperiments),1)))),np.linalg.inv(TBase0Rot))[:,:3]
    PijTest[:,referencePointIndex,:] = np.dot(np.hstack((Items[referencePointsItemsPrefix+str(referencePointIndex)][indexTest,:3],np.ones((len(indexTest),1)))),np.linalg.inv(TBase0Rot))[:,:3]

Jointi = Items["jointsAngles"][indexExperiments]
JointiTest = Items["jointsAngles"][indexTest]

### STEP 1 - BUILD IDENTIFICATION MATRIX
#Pij = PBase + PRobot(qi,pi0) + [~PhiBase]*Probot(qi,pi0) + Jpi(p)(qi)*deltapi + [~PhiBase]*Jpi(p)(qi)*deltapi + RRobot(qi,pi0)*PToolj + [~PhiBase]*RRobot(qi,pi0)*PToolj + [~Jpi(r)(qi)*deltapi]*PToolj + [~PhiBase]*[~Jpi(r)(qi)*deltapi]*PToolj
# => Pij = IdMatrix * (PBase PhiBase deltapi 
#               PhiBaseX*deltapi PhiBaseY*deltapi PhiBaseZ*deltapi 
#               PToolj PhiBaseX*PToolj PhiBaseY*PToolj PhiBaseZ*PToolj 
#               PTooljX*deltapi PTooljY*deltapi PTooljZ*deltapi 
#               PhiBaseX*PTooljX*deltapi PhiBaseY*PTooljY*deltapi PhiBaseZ*PTooljZ*deltapi
#               PhiBaseX*PTooljY*deltapi PhiBaseX*PTooljZ*deltapi 
#               PhiBaseY*PTooljX*deltapi PhiBaseY*PTooljZ*deltapi
#               PhiBaseZ*PTooljX*deltapi PhiBaseZ*PTooljY*deltapi) + PRobot(qi,pi0)

N_ID_PARAMETERS = 3 + 3 + N_DELTAPI + 3*N_DELTAPI + N_REFERENCEPOINTS*3 + 3*N_REFERENCEPOINTS*3 + 3*N_DELTAPI*N_REFERENCEPOINTS + N_DELTAPI*N_REFERENCEPOINTS*9

def skewVector(v):
    return(np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]))

def skewMatrix(M):
    L = np.shape(M)[1]
    return(np.vstack((np.hstack((np.zeros(L),-M[2],M[1])),
                      np.hstack((M[2],np.zeros(L),-M[0])),
                      np.hstack((-M[1],M[0],np.zeros(L))))))

def doubleSkewMatrix(M):
    L = np.shape(M)[1]
    return(np.vstack((np.hstack((np.zeros(L),M[0],M[0],np.zeros(2*L),-M[1],np.zeros(L),-M[2],np.zeros(L))),
                      np.hstack((M[1],np.zeros(L),M[1],-M[0],np.zeros(4*L),-M[2])),
                      np.hstack((M[2],M[2],np.zeros(2*L),-M[0],np.zeros(L),-M[1],np.zeros(2*L))))))

def IdMatrix(q,pi,jTool):
    TRobot = T(q,pi)
    RRobot = TRobot[:3,:3]
    PRobot = TRobot[:3,-1]
    J = Jpi(q,pi)

    IdMatrix = np.zeros((3,N_ID_PARAMETERS))
    IdMatrix[:,:3] = np.eye(3)
    IdMatrix[:,3:6] = -skewVector(PRobot)
    IdMatrix[:,6:6 + N_DELTAPI] = J[:3]
    IdMatrix[:,6 + N_DELTAPI:6 + 4*N_DELTAPI] = -skewMatrix(J[:3])

    IdMatrix[:,6 + 4*N_DELTAPI + 3*jTool:6 + 4*N_DELTAPI + 3*jTool + 3] = RRobot 
    IdMatrix[:,6 + 4*N_DELTAPI + 3*N_REFERENCEPOINTS + 9*jTool:6 + 4*N_DELTAPI + 3*N_REFERENCEPOINTS + 9*jTool + 9] = -skewMatrix(RRobot)
    IdMatrix[:,6 + 4*N_DELTAPI + 3*N_REFERENCEPOINTS + 9*N_REFERENCEPOINTS + 3*N_DELTAPI*jTool:6 + 4*N_DELTAPI + 3*N_REFERENCEPOINTS + 9*N_REFERENCEPOINTS + 3*N_DELTAPI*jTool + 3*N_DELTAPI] = skewMatrix(J[3:])
    IdMatrix[:,6 + 4*N_DELTAPI + 3*N_REFERENCEPOINTS + 9*N_REFERENCEPOINTS + 3*N_DELTAPI*N_REFERENCEPOINTS + N_DELTAPI*9*jTool:6 + 4*N_DELTAPI + 3*N_REFERENCEPOINTS + 9*N_REFERENCEPOINTS + 3*N_DELTAPI*N_REFERENCEPOINTS + N_DELTAPI*9*jTool + N_DELTAPI*9] = doubleSkewMatrix(J[3:])

    return(IdMatrix)

M = np.empty((3,N_ID_PARAMETERS))
P = np.empty((1,3))

for q in Jointi:
    for j in range(N_REFERENCEPOINTS):
        M = np.vstack((M,IdMatrix(q,pi0,j)))
        P = np.vstack((P, T(q,pi0)[:3,-1]))

delta = np.dot(np.linalg.pinv(M),P-Pij.flatten())