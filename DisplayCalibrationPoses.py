from GeometricModel import *
from CalibrationCSVParsing import *
import matplotlib.pyplot as plt

import yaml

from AnalyticGeometricModel import Jfunc

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

### RETREIVE NOMINAL PARAMETERS
TBase0 = TBaseN

TTool0 = np.empty((len(TTool0_marker),4,4))
for j in range(len(TTool0_marker)):
    TTool0[j] = np.dot(TToolN,TTool0_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

def plotFrame(transformationMatrix, ax = None):
    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([1,0,0,0]))[:3],color="r",length=0.1,normalize=True)
    ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([0,1,0,0]))[:3],color="g",length=0.1,normalize=True)
    ax.quiver(*transformationMatrix[:3,-1],*np.dot(transformationMatrix,np.array([0,0,1,0]))[:3],color="b",length=0.1,normalize=True)
    ax.scatter(*transformationMatrix[:3,-1],s=5,color="k")
    return(ax)

CalibrationConfigurations = np.empty((0,0))
CalibrationPoses = np.empty((0,3))

print("Loading calibration poses...")
with open("CalibrationConfigurations_" + argv[1] + "_" + argv[2] + "_" + argv[3] + ".yaml","r") as file:
    CalibrationConfigurations = np.array(yaml.load(file, Loader=yaml.FullLoader)["calibrationConfigurations"]).reshape((-1,JOINTS_NUMBER))
print("Done !")

print('Number of configurations : ' + str(len(CalibrationConfigurations)))

identifiability = input("Compute identifiability score ? y/n")

if(identifiability == "y" or identifiability == "Y"):
    print("Identifiability score :")
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    SVD = np.linalg.svd(J)[1]

    print("Singular values : " + str(SVD))

    score = SVD[0]/SVD[-1]
    print("score : " + str(score))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for q in CalibrationConfigurations:
    transformationMatrix = T(q,pi0)
    CalibrationPoses = np.vstack((CalibrationPoses,transformationMatrix[:3,-1]))
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