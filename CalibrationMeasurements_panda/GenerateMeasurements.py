import csv
from random import sample,uniform
import numpy as np

from importlib import import_module
from sys import argv

from GeometricModel import T

from optiTrack_parameters import TTool0_optiTrack

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

### Create deltapi0
deltapi0 = np.zeros(len(pi0))
parametersCounter = 0

for i in range(JOINTS_NUMBER):

    #Joint offset
    if (JOINTS_OFFSET[i]==1):
        deltapi0[parametersCounter] = uniform(-1,1)*(np.pi/180)
        parametersCounter+=1

    #Link translation
    for translation in LINKS_TRANSLATIONS[i]:
        deltapi0[parametersCounter] = uniform(-1,1)*0.001
        parametersCounter+=1

    #Link rotation
    for rotation in LINKS_ROTATIONS[i]:
        deltapi0[parametersCounter] = uniform(-1,1)*(np.pi/180)
        parametersCounter+=1

### Create Base error

PhiBase = [uniform(-1,1)*(np.pi/180),uniform(-1,1)*(np.pi/180),uniform(-1,1)*(np.pi/180)]
DeltaBase = [uniform(-1,1)*0.001,uniform(-1,1)*0.001,uniform(-1,1)*0.001]
TBase = np.array([[0, -PhiBase[2], PhiBase[1], DeltaBase[0]],
                [PhiBase[2], 0, -PhiBase[0], DeltaBase[1]],
                [-PhiBase[1], PhiBase[0], 0, DeltaBase[2]],
                [0,0,0,1]])

### Create markers error

TMarker = []
for marker in TTool0_optiTrack:
    TMarker.append(np.array([[0, 0, 0, uniform(-1,1)*0.001],
                                [0, 0, 0, uniform(-1,1)*0.001],
                                [0, 0, 0, uniform(-1,1)*0.001],
                                [0,0,0,1]]))

### Create joints angles

#Default number of configurations
N = 100

try:
    N = int(argv[2])
except:
    print("Switching to default value : N=100")

print("Creating N = " + str(N) + " measurements")

Q = [sample(list(np.linspace(0,2*np.pi,1000)),JOINTS_NUMBER) for i in range(N)]

with open('GeneratedCalibration_' + argv[1] + '.csv', 'w') as f:
    writer = csv.writer(f)

    for q in Q:
        Tdelta = T(q,pi0+deltapi0,TBaseN + TBase,TToolN)

        for i,marker in enumerate(TTool0_optiTrack):
            TdeltaM = np.dot(Tdelta,marker + TMarker[i]) 
            writer.writerow(TdeltaM[:3,-1])

        writer.writerow(q)
        writer.writerow(['---'])

with open('GeneratedErrors' + argv[1] + '.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['deltapi0'])
    writer.writerow(deltapi0)
    writer.writerow(['TBase'])
    for row in TBase:
        writer.writerow(row)
    for i,marker in enumerate(TTool0_optiTrack):
        writer.writerow(['TMarker' + str(i+1)])
        for row in TMarker[i]:
            writer.writerow(row)
