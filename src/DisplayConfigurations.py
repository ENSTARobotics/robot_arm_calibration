#!/usr/bin/python3

#System packages
from importlib import import_module
import sys
import os
import yaml
import glob

#Utility packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

### GET CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/"
sys.path.append(configPath)

### RETRIEVE PARAMETERS FROM DIRECTORY NAME
import re
currentParameters = np.array(re.split('/|_',os.getcwd()))
currentParameters = currentParameters[np.where(currentParameters == "calibration")[0][0] + 1:]

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
from AnalyticGeometricModel import Jfunc
from NumericGeometricModel import T

### USEFULL FUNCTION
def plotFrame(transformationMatrix, ax = None):
    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([1,0,0,0]))[:3],color="r",length=0.1,normalize=True)
    ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([0,1,0,0]))[:3],color="g",length=0.1,normalize=True)
    ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([0,0,1,0]))[:3],color="b",length=0.1,normalize=True)
    ax.scatter(*transformationMatrix[:3,-1],s=5,color="k")
    return(ax)

### RETREIVE NOMINAL PARAMETERS
TBase0 = TBaseN

TTool0 = np.empty((len(TToolN_marker),4,4))
for j in range(len(TToolN_marker)):
    TTool0[j] = np.dot(TFlangeN,TToolN_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

### LOADING CALIBRATION POSES

print("Loading configurations...")
fileName = glob.glob("*Configurations.yaml")[0]
print(fileName)
with open(fileName,"r") as file:
    Configurations = yaml.load(file, Loader=yaml.FullLoader)
    Configurations = np.array(Configurations[list(Configurations.keys())[0]]).reshape((-1,JOINTS_NUMBER))
print("Done !")

print('Number of configurations : ' + str(len(Configurations)))

identifiability = input("Compute identifiability score ? y/n")

if(identifiability == "y" or identifiability == "Y"):
    print("Identifiability score :")
    J = np.vstack([Jfunc(parameters0,q) for q in Configurations])
    SVD = np.linalg.svd(J)[1]

    print("Singular values : " + str(SVD))

    score = SVD[0]/SVD[-1]
    print("score : " + str(score))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Poses = np.empty((0,3))
for q in Configurations:
    transformationMatrix = T(q,pi0)
    Poses = np.vstack((Poses,transformationMatrix[:3,-1]))
    plotFrame(transformationMatrix,ax)

extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
sz = extents[:,1] - extents[:,0]
centers = np.mean(extents, axis=1)
maxsize = max(abs(sz))
r = maxsize/2
for ctr, dim in zip(centers, 'xyz'):
    getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
plt.legend()
plt.show()