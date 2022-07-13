#!/usr/bin/python3

#System packages
from importlib import import_module
import sys
import os

#Utility packages
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.get_cmap('hsv',100)
from scipy.spatial.transform import Rotation as R

### GET CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/"
sys.path.append(configPath)

### RETRIEVE PARAMETERS FROM DIRECTORY NAME
import re
currentParameters = np.array(re.split('/|_',os.getcwd()))
currentParameters = currentParameters[np.where(currentParameters == "measurements")[0][0] + 1:]

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(currentParameters[0]+"_parameters")

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
    module = import_module(currentParameters[1]+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID MARKER NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

#Custom packages
from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation
from AnalyticGeometricModel import Pfunc

### RETRIEVE MEASUREMENTS DATA
itemsNames = robotBaseItemsNames + markerItemsNames + ["jointsAngles"]
Items = calibrationCSVParsing(itemsNames,"Calibration.csv")
TRobotBase = robotBaseTransformation(Items)

N_MEASUREMENTS = len(Items["jointsAngles"])

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS
TBase0 = np.dot(TRobotBase,TBaseN)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TToolN,TTool0_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

def plotMarkersError(configurationIndex):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i,item in enumerate(markerItemsNames):
        color = cmap(int(i*100/N_REFERENCEPOINTS))

        #Plot measured markers positions
        Mmeasured = Items[item][configurationIndex]
        ax.scatter(*Mmeasured,color = color,label="Marker " + str(i))

        #Plot computed markers positions
        Mcomputed = Pfunc(parameters0,Items["jointsAngles"][configurationIndex])
        ax.scatter(*Mcomputed,color = color, marker='*')

        print("Marker " + str(i) + " - Configuration " + str(configurationIndex))
        print("Measured position : " + str(Mmeasured))
        print("Computed position : " + str(Mcomputed))

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Markers computed and measured positions - Configuration : " + str(configurationIndex))

    plt.legend()
    plt.show()

def plotError():
    _, ax = plt.subplots(nrows=N_REFERENCEPOINTS+1)

    Delta = np.empty((N_REFERENCEPOINTS,N_MEASUREMENTS,3))
    configurationsRange = np.arange(1,N_MEASUREMENTS+1)
    labels = ["X","Y","Z"]
    
    for i,item in enumerate(markerItemsNames):
        for j in range(N_MEASUREMENTS):
            Delta[i,j,:] = Items[item][j] - Pfunc(parameters0,Items["jointsAngles"][j])
        for k,l in enumerate(labels):
            ax[i].plot(configurationsRange,Delta[i,:,k],label=l)
        ax[i].set_title("Marker " + str(i+1))

    AvgDelta = np.average(Delta,axis=0)
    for k,l in enumerate(labels):
        ax[-1].plot(configurationsRange,AvgDelta[:,k],label=l)
    ax[-1].set_title("Mean value")

    plt.legend()
    plt.show()

plotMarkersError(1)
plotMarkersError(10)
plotMarkersError(20)
plotMarkersError(30)
plotError()

    


