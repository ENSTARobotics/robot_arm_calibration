from GeometricModel import *
from CalibrationCSVParsing import *
import matplotlib.pyplot as plt

import yaml

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
with open("CalibrationConfigurations_" + argv[1] + "_" + argv[2] + ".yaml","r") as file:
    CalibrationConfigurations = np.array(yaml.load(file, Loader=yaml.FullLoader)["calibrationConfigurations"]).reshape((-1,JOINTS_NUMBER))
print("Done !")

print('Number of configurations : ' + str(len(CalibrationConfigurations)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for q in CalibrationConfigurations:
    transformationMatrix = T(q,pi0)
    CalibrationPoses = np.vstack((CalibrationPoses,transformationMatrix[:3,-1]))
    plotFrame(transformationMatrix,ax)

"""
TestConfigurations = np.array([[-2.66963,	-0.768568,	-0.271546,	-2.9705,	-1.50095,	3.08698],
                               [1.41736,	-1.13054,	1.02873,	-2.52073,	1.94506,	2.13568],
                               [-0.589849,	-1.0708,	-1.19791,	-0.837837,	2.76945,	-1.70423],
                               [-0.443319,	-2.13328,	-0.578341,	2.66722,	-1.83742,	-1.72546],
                               [-2.54072,	-1.86348,	0.821925,	-1.35056,	1.19397,	-3.09711],
                               [0.533859,	-1.6264,	0.8308,	    -2.38834,	-0.0458506,	-2.69648],
                               [0.483007,	-1.03659,	-0.923424,	0.475009,	1.94361,	-0.243918],
                               [-1.03259,	-2.52476,	-1.16237,	1.667,	    1.23989,	-0.411907],
                               [1.06684,	-0.178461,	-0.0667104,	-1.32708,	1.3323,  	1.57639]])

minimumDistancesC = []
minimumDistancesP = []

for i,q in enumerate(TestConfigurations):
    print("Test configuration :")
    print(q)
    Distances = np.linalg.norm(CalibrationConfigurations-q,axis=1)
    indexMin = np.argmin(Distances)
    print("Closest calibration configuration : ")
    print(CalibrationConfigurations[indexMin])
    print("Corresponding distance :")
    print(Distances[indexMin])
    minimumDistancesC.append(Distances[indexMin])

    p = T(q,pi0)[:3,-1]
    print("Test pose :")
    print(p)
    Distances = np.linalg.norm(CalibrationPoses-p,axis=1)
    indexMin = np.argmin(Distances)
    print("Closest calibration configuration : ")
    print(CalibrationPoses[indexMin])
    print("Corresponding distance :")
    print(Distances[indexMin])
    minimumDistancesP.append(Distances[indexMin])

    ax.scatter(*p,label=str(i+1))
"""

extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
sz = extents[:,1] - extents[:,0]
centers = np.mean(extents, axis=1)
maxsize = max(abs(sz))
r = maxsize/2
for ctr, dim in zip(centers, 'xyz'):
    getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
plt.legend()
plt.show()