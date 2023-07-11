
import csv
import sys
from collections import OrderedDict
import numpy as np

from scipy.spatial.transform import Rotation as R

import os
from importlib import import_module

### RETREIVE ROBOT AND MARKER NAMES
import re
currentParameters = np.array(re.split('/|_',os.getcwd()))
currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
robotName = currentParameters[0]
markerName = currentParameters[1]

### GET ROBOT CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/nominalModels/" + robotName + "/"
sys.path.append(configPath)

print(configPath)

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(robotName+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID ROBOT NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

### GET MARKER CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/nominalModels/markers/"
sys.path.append(configPath)

### IMPORT MARKER PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(markerName+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID MARKER NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

def csvRowToArray(csvRow):
    arrayRow = []
    for csvItem in csvRow:
        csvItem.replace(',','.')
        arrayItem = float(csvItem)
        arrayRow.append(arrayItem)
    return(np.array(arrayRow))

def calibrationCSVParsing(itemsNames,calibrationCSVPath):

    Items = OrderedDict.fromkeys(itemsNames,np.empty((0,0)))

    with open(calibrationCSVPath, "r") as csvfile:
        rowCounter = 0
        rowReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in rowReader:
            if(row[0]=="---"):
                rowCounter = 0
            else:
                currentRow = csvRowToArray(row)
                if(np.shape(Items[itemsNames[rowCounter]]) == (0,0)):
                    Items[itemsNames[rowCounter]] = np.array([currentRow])
                else:
                    Items[itemsNames[rowCounter]] = np.append(Items[itemsNames[rowCounter]],np.array([currentRow]),axis=0)
                rowCounter+=1

    return(Items)

def filteredMean(array, axis = 0):
    if(len(array.shape) == 1 or (len(array.shape) == 2 and (array.shape[0] == 1 or array.shape[1] == 1))):
        filteredArray = array[np.where(np.abs(array - np.mean(array)) < 0.0001 + np.std(array))]

        return(np.mean(filteredArray))

    elif(len(array.shape) == 2 and array.shape[0] != 1 or array.shape[1] != 1):
        filteredArray = np.empty(array.shape[(axis+1)%2])
        if(axis == 0):
            splittedArray = np.hsplit(array,array.shape[1])
            for i,item in enumerate(splittedArray):
                filteredArray[i] = filteredMean(item)
        elif(axis == 1):
            splittedArray = np.vsplit(array,array.shape[0])
            for i,item in enumerate(splittedArray):
                filteredArray[i] = filteredMean(item)
        else:
            raise IndexError
        return(filteredArray)

    else:
        raise NotImplementedError

def modAngle(angle):
    try:
        output = []
        for item in angle:
            output.append(modAngle(item))
        return(np.array(output))

    except TypeError:
        if(angle <= -np.pi):
            return(modAngle(angle + 2*np.pi))
        elif(angle > np.pi):
            return(modAngle(angle - 2*np.pi))
        else:
            return(angle)

def planeProjection(point,planeNormal,planePoint):
    v = point - planePoint
    dist = np.dot(planeNormal,v)
    return(point - dist*planeNormal)

if(robotName == "ur10e"):
    robotBaseOffset = np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,-0.00565],
                                [0,0,0,1]])
    robotBaseItemsNames = ["robotBase1","robotBase2","robotBase3","robotBase4","robotBase5","robotBase6","robotBase7"]

if(robotName == "panda"):
    robotBaseOffset = np.array([[0,-1,0,0],
                                [1,0,0,(0.120/2) - 0.020 + 0.011],
                                [0,0,1,0.0217-0.00565],
                                [0,0,0,1]])
    robotBaseItemsNames = ["robotBase1","robotBase2","robotBase3","robotBase4","robotBase5","robotBase6","robotBase7"]

def robotBaseTransformation(Items):
    
    # ________________________
    #|(6)                  (1)|
    #|                 (7)    |
    #|                        |
    #|           _|_ (2)      |  
    #|      (5)   |           |      Y
    #|                        |      |
    #|                        |     -|-- X
    #|(4)__________________(3)|          
    #           
        
    #Retrieve the robot base markers positions
    robotBase = np.array([filteredMean(Items[item],axis=0) for item in robotBaseItemsNames])
    baseCentroid = np.mean(robotBase[:-1],axis=0)

    #Find the plane normal
    z = np.dot(np.linalg.pinv(robotBase),np.ones(len(robotBase)))
    z/=np.linalg.norm(z)
    if(z[2] < 0):
        z*=-1

    #Find the other axis
    if(robotName == "ur10e"):
        x = (planeProjection(robotBase[-1],z,baseCentroid) - planeProjection(robotBase[5],z,baseCentroid))  #LEGACY FOR UR10e
    else:
        x = (planeProjection(robotBase[0],z,baseCentroid) - planeProjection(robotBase[5],z,baseCentroid))
    #x = (robotBase[0] - robotBase[5]) #OLD WAY (WRONG, BUT LEGACY)
    x = x/np.linalg.norm(x)
    y = np.cross(z,x)

    #Retreive robot base transformation
    
    #Interpolated frame (C) to OptiTrack reference frame (W)
    TWC = np.eye(4)
    TWC[:3,:3] = np.array([x,y,z]).T
    TWC[:3,-1] = baseCentroid

    #Robot base frame (R) to interpolated frame (C)
    TCR = robotBaseOffset

    #Robot base frame(R) to Optitrack reference frame (W)
    robotBaseTransformation = np.dot(TWC,TCR)
   
    return(np.linalg.inv(robotBaseTransformation))

sys.path.append("/home/demo1/Desktop/robot_arm_calibration/scripts/")
from AnalyticGeometricModel import Pfunc,Jfunc,size

TBase0 = TBaseN

TTool0 = np.empty((N_REFERENCEPOINTS,4,4))
for j in range(N_REFERENCEPOINTS):
    TTool0[j] = np.dot(TFlangeN,TToolN_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

markerItemsNames = ["optiTrackPose" + str(i) for i in range(7)]
itemsNames = robotBaseItemsNames + markerItemsNames + ["jointsAngles"]

Items = calibrationCSVParsing(itemsNames,sys.argv[1])
robotBase = robotBaseTransformation(Items)

tmp = []
with open("new_" + sys.argv[1], "w") as csvfile:
    output = csv.writer(csvfile, delimiter=',', quotechar='|')
    for i in range(len(Items['jointsAngles'])):
        #print("Comparaison " + str(i))
        for j,item in enumerate(markerItemsNames):
            P = np.round(np.dot(robotBase,np.append(Items[item][i],1))[:3],5)
            output.writerow(P)
            #print(P)
            #print(Pfunc(parameters0,Items["jointsAngles"][i])[3*j:3*j+3].T[0])
            tmp.append(P-Pfunc(parameters0,Items["jointsAngles"][i])[3*j:3*j+3].T[0])
        output.writerow(Items['jointsAngles'][i])
        output.writerow(["---"])
        #input()

tmp = np.array(tmp)
print(np.average(tmp,axis=0))
input()
import subprocess
subprocess.call(["mv", sys.argv[1], "old_" + sys.argv[1]])
subprocess.call(["mv", "new_" + sys.argv[1], sys.argv[1]])