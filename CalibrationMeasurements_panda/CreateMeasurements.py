import csv
from random import sample,uniform
import numpy as np

from importlib import import_module
from sys import argv

from GeometricModel import T

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

### Create reference points

TTool = np.empty((3,4,4))

TTool[0] = np.eye(4)
TTool[0][0,-1] = 0.1

TTool[1] = np.eye(4)
TTool[1][1,-1] = 0.1

TTool[2] = np.eye(4)
TTool[2][2,-1] = 0.1

### Create joints angles

N = 100
Q = [sample(list(np.linspace(0,2*np.pi,1000)),JOINTS_NUMBER) for i in range(N)]

with open('GeneratedCalibration.csv', 'w') as f:
    writer = csv.writer(f)

    for q in Q:
        Tdelta = T(q,pi0+deltapi0)

        Tdelta0 = np.dot(Tdelta,TTool[0]) 
        Tdelta1 = np.dot(Tdelta,TTool[1])
        Tdelta2 = np.dot(Tdelta,TTool[2])

        writer.writerow(Tdelta0[:3,-1])
        writer.writerow(Tdelta1[:3,-1])
        writer.writerow(Tdelta2[:3,-1])
        writer.writerow(q)
        writer.writerow(['---'])

with open('Generateddeltapi0.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(deltapi0)
