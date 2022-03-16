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

### STEP 1 - PBase, RBase & PToolj

def Ai(q,pi):
    Ti = T(q,pi)
    PRoboti = Ti[:3,-1]
    PRobotiTildeT = np.array([[0,PRoboti[2],-PRoboti[1]],
                         [-PRoboti[2],0,PRoboti[0]],
                         [PRoboti[1],-PRoboti[0],0]])
    RRoboti = Ti[:3,:3]

    Ai = np.hstack([np.vstack([np.eye(3)]*N_REFERENCEPOINTS),np.vstack([PRobotiTildeT]*N_REFERENCEPOINTS),np.kron(np.eye(N_REFERENCEPOINTS,dtype=int),RRoboti)])
    return(Ai)

def step1(pi, mode = "L"):
    A = np.vstack([Ai(Jointi[i],pi) for i in range(N_EXPERIMENTS)])
    deltaP = np.hstack([(Pij[i,:] - T(Jointi[i],pi)[:3,-1]).flatten() for i in range(N_EXPERIMENTS)])

    output = np.zeros(np.shape(A)[1])

    if(mode == "L"):
        output = np.dot(np.linalg.pinv(A),deltaP)

    elif(mode == "O"):
        def f(beta,x):
            return(np.dot(x.T,beta))

        linearModel = Model(f)
        data = Data(A.T,deltaP)
        odr = ODR(data,linearModel,beta0=output)
        out = odr.run()
        output = out.beta

    else:
        raise NotImplementedError("Unknown regression mode")

    TBase = np.eye(4) + np.array([[0,-output[5],output[4],output[0]],
                                  [output[5],0,-output[3],output[1]],
                                  [-output[4],output[3],0,output[2]],
                                  [0,0,0,0]])

    TTool = np.empty((N_REFERENCEPOINTS,4,4))
    for j in range(N_REFERENCEPOINTS):
        TTool[j] = np.eye(4)
        TTool[j][:3,-1] = np.dot(TBase[:3,:3].T,output[6+j*3:9+j*3])

    return(TBase,TTool)

### STEP 2 - pi

def step2(TBase,TTool, mode = "L"):
    J = np.vstack([Jpi(Jointi[i],pi0,TBase,TTool[j])[:3,:] for i in range(N_EXPERIMENTS) for j in range(N_REFERENCEPOINTS)])
    deltaP = np.hstack([Pij[i,j] - T(Jointi[i],pi0,TBase,TTool[j])[:3,-1] for i in range(N_EXPERIMENTS) for j in range(N_REFERENCEPOINTS)])

    deltapi = np.zeros(np.shape(J)[1])

    if(mode == "L"):
        deltapi = np.dot(np.linalg.pinv(J),deltaP)

    elif(mode == "O"):
        def f(beta,x):
            return(np.dot(x.T,beta))

        linearModel = Model(f)
        data = Data(J.T,deltaP)
        odr = ODR(data,linearModel,beta0=deltapi)
        out = odr.run()
        deltapi = out.beta

    else:
        raise NotImplementedError("Unknown regression mode")
    
    return(pi0+deltapi.T)

### COMPUTE ERROR

def error(TBase,TTool,pi):
    return(np.sum(np.sum(np.array([np.array([np.linalg.norm(PijTest[-i-1,j] - T(JointiTest[-i-1],pi,TBase,TTool[j])[:3,-1]) for j in range(N_REFERENCEPOINTS)]) for i in range(N_TESTS)]),axis=0),axis=0))

### MAIN LOOP

pi = pi0
TBase = TBase0
TTool = TTool0

e = error(TBase,TTool,pi)

print("--- INITIAL CONFIGURATION ---")

print("TBase = ")
display(TBase)

for i,M in enumerate(TTool0):
    print("TTool" + str(i) + " = ")
    display(M)

print("pi0 = " + str(pi))
print("Error = " + str(e) + " m")

N = 10
Errors = [e]

for i in range(N):
    print("--- ITERATION " + str(i+1) +" ---")
    TBase,TTool = step1(pi,"L")
    pi = step2(TBase,TTool,"L")

    e = error(TBase,TTool,pi)
    Errors.append(e)

    print("TBase = ")
    display(TBase)

    for j,M in enumerate(TTool):
        print("TTool" + str(j+1) + " = ")
        display(M)

    print("deltapi0 = " + str(pi-pi0))
    print("Error = " + str(e) + " m")

plt.figure()
plt.plot(np.arange(0,N+1,1),Errors)
plt.xlabel(r"Iterations")
plt.ylabel(r"$\Sigma~|X_{mes} - X_{comp}|~$(m)")
plt.show()




