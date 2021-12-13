#!/usr/bin/python3

from CalibrationCSVParsing import calibrationCSVParsing,robotBaseTransformation
from GeometricModel import *
from optiTrack_parameters import *

from importlib import import_module
from sys import argv

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

Items = calibrationCSVParsing(itemsNames,"Calibration1.csv")

referencePointsIndices = [int(key[-1]) for key,_ in Items.items() if referencePointsItemsPrefix in key]
N_REFERENCEPOINTS = len(referencePointsIndices)

N_CONFIGURATIONS = len(Items[itemsNames[0]])

### ROBOT BASE AND REFERENCE POINTS TRANSFORMATIONS ESTIMATIONS

TBase0 = np.dot(robotBaseTransformation(Items,robotBaseMarkersOffset,robotBasePlaneOffset,robotBaseItemsNames),TBaseN)
TBase0Inv = np.linalg.inv(TBase0)

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TToolN,TTool0_optiTrack[j])

def plotMarkersError(configurationIndex,pi=pi0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(N_REFERENCEPOINTS):
        color = cmap(int(i*100/N_REFERENCEPOINTS))

        #Plot measured markers positions
        Mmeasured = np.dot(TBase0Inv,np.hstack((Items[referencePointsItemsPrefix+str(i)][configurationIndex],1)))[:3]
        ax.scatter(Mmeasured[0],Mmeasured[1],Mmeasured[2],color = color,label="Marker " + str(i))

        #Plot computed markers positions
        Mcomputed = T(Items["jointsAngles"][configurationIndex],pi,np.eye(4),TTool0[i])[:3,-1]
        ax.scatter(Mcomputed[0],Mcomputed[1],Mcomputed[2],color = color, marker='*')

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

def plotError(pi=pi0):
    fig, ax = plt.subplots(nrows=N_REFERENCEPOINTS+1)

    Delta = np.empty((N_CONFIGURATIONS,N_REFERENCEPOINTS,3))
    N = np.arange(1,N_CONFIGURATIONS+1)
    labels = ["X","Y","Z"]
    
    for j in range(N_REFERENCEPOINTS):
        for i in range(N_CONFIGURATIONS):
            Delta[i,j,:] = np.dot(TBase0Inv,np.hstack((Items[referencePointsItemsPrefix+str(j)][i],1)))[:3] - T(Items["jointsAngles"][i],pi,np.eye(4),TTool0[j])[:3,-1]
        for k,l in enumerate(labels):
            ax[j].plot(N,Delta[:,j,k],label=l)
        ax[j].set_title("Marker " + str(j))

    AvgDelta = np.average(Delta,axis=1)
    for k,l in enumerate(labels):
        ax[-1].plot(N,AvgDelta[:,k],label=l)
    ax[-1].set_title("Mean value")

    plt.legend()
    plt.show()

plotMarkersError(1)
plotMarkersError(10)
plotMarkersError(20)
plotMarkersError(30)
plotError()

    


