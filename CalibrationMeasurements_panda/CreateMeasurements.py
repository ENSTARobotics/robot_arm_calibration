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
        deltapi0[parametersCounter] = uniform(-1,1)*np.pi*0.1
        parametersCounter+=1

    #Link translation
    for translation in LINKS_TRANSLATIONS[i]:
        deltapi0[parametersCounter] = uniform(-1,1)*0.1
        parametersCounter+=1

    #Link rotation
    for rotation in LINKS_ROTATIONS[i]:
        deltapi0[parametersCounter] = uniform(-1,1)*np.pi*0.1
        parametersCounter+=1

### Create joints angles

#Default number of configurations
N = 100

try:
    N = int(argv[2])
except:
    print("Switching to default value : N=100")

print("Creating N = " + str(N) + " measurements")

Q = [sample(list(np.linspace(0,2*np.pi,1000)),JOINTS_NUMBER) for i in range(N)]

with open('GeneratedCalibration.csv', 'w') as f:
    writer = csv.writer(f)

    for q in Q:
        Tdelta = T(q,pi0+deltapi0,TBaseN,TToolN)

        for marker in TTool0_optiTrack:
            TdeltaM = np.dot(Tdelta,marker) 
            writer.writerow(TdeltaM[:3,-1])

        writer.writerow(q)
        writer.writerow(['---'])

with open('Generateddeltapi0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(deltapi0)
