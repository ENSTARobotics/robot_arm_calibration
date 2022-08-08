#!/usr/bin/python3

#System packages
from importlib import import_module
import sys
import os
import csv

#Utility packages
import numpy as np
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
from NumericGeometricModel import T

### Create deltapi
deltapi = np.zeros(len(pi0))
parametersCounter = 0

for i in range(JOINTS_NUMBER):

    #Joint offset
    if (JOINTS_OFFSET[i]==1):
        deltapi[parametersCounter] = np.random.uniform(-1,1)*(np.pi/180)
        parametersCounter+=1

    #Link translation
    for translation in LINKS_TRANSLATIONS[i]:
        deltapi[parametersCounter] = np.random.uniform(-1,1)*0.001
        parametersCounter+=1

    #Link rotation
    for rotation in LINKS_ROTATIONS[i]:
        deltapi[parametersCounter] = np.random.uniform(-1,1)*(np.pi/180)
        parametersCounter+=1

### Create Base error

RBase = R.from_euler('xyz',[np.random.uniform(-1,1)*(np.pi/180),np.random.uniform(-1,1)*(np.pi/180),np.random.uniform(-1,1)*(np.pi/180)]).as_matrix()
PBase = np.array([[np.random.uniform(-1,1)*0.001],[np.random.uniform(-1,1)*0.001],[np.random.uniform(-1,1)*0.001]])
TBase = np.vstack((np.hstack((RBase,PBase)),np.array([[0,0,0,1]])))
#TBase = np.eye(4)

### Create markers error

TMarker = []
for marker in TToolN_marker:
    TMarker.append(np.array([[1, 0, 0, np.random.uniform(-1,1)*0.001],
                             [0, 1, 0, np.random.uniform(-1,1)*0.001],
                             [0, 0, 1, np.random.uniform(-1,1)*0.001],
                             [0,0,0,1]]))
    #TMarker.append(np.eye(4))

### Create joints angles

#Default number of configurations
N = 100

try:
    N = int(sys.argv[3])
except:
    print("Switching to default value : N = 100")

print("Creating N = " + str(N) + " measurements")

Q = [np.random.sample(list(np.linspace(0,2*np.pi,1000)),JOINTS_NUMBER) for i in range(N)]

with open('GeneratedMeasurements.csv', 'w') as f:
    writer = csv.writer(f)

    for q in Q:
        Tdelta = T(q,pi0+deltapi,np.dot(TBaseN,TBase),TFlangeN)

        for i,marker in enumerate(TToolN_marker):
            TdeltaM = np.dot(Tdelta,np.dot(marker,TMarker[i])) 
            writer.writerow(TdeltaM[:3,-1])

        writer.writerow(q)
        writer.writerow(['---'])

with open('GeneratedErrors.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['pi'])
    writer.writerow(pi0 + deltapi)
    writer.writerow(['TBase'])
    for row in np.dot(TBaseN,TBase):
        writer.writerow(row)
    for i,marker in enumerate(TToolN_marker):
        writer.writerow(['TTool' + str(i+1)])
        for row in np.dot(TFlangeN,np.dot(marker,TMarker[i])):
            writer.writerow(row)
