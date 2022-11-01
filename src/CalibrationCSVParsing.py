#!/usr/bin/python3

#System packages
from importlib import import_module
import sys
import os
import csv
from collections import OrderedDict

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

### GET CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/"
sys.path.append(configPath)

if __name__ == "__main__":
    robotName = sys.argv[1]
else:
    import re
    currentParameters = np.array(re.split('/|_',os.getcwd()))
    currentParameters = currentParameters[np.where(currentParameters == "calibration")[0][0] + 1:]
    robotName = currentParameters[0]

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(robotName + "_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID ROBOT NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

### USEFULL FUNCTIONS

def csvRowToArray(csvRow):
    arrayRow = []
    for csvItem in csvRow:
        csvItem.replace(',','.')
        arrayItem = float(csvItem)
        arrayRow.append(arrayItem)
    return(np.array(arrayRow))

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

### CSV PARSING

def planeProjection(point,planeNormal,planePoint):
    v = point - planePoint
    dist = np.dot(planeNormal,v)
    return(point - dist*planeNormal)

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
   
    return(robotBaseTransformation)

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

def errorsCSVParsing(errorsCSVPath):

    parametersDict = {}
    parametersDict["TTool"] = []

    with open(errorsCSVPath, "r") as csvfile:
        rowReader = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
        for i,row in enumerate(rowReader):
            if(row[0] == "pi"):
                parametersDict["pi"] = csvRowToArray(rowReader[i+1])
            elif(row[0] == "TBase"):
                parametersDict["TBase"] = np.array([csvRowToArray(rowReader[i+k]) for k in range(1,5)])
            elif("TTool" in row[0]):
                parametersDict["TTool"].append(np.array([csvRowToArray(rowReader[i+k]) for k in range(1,5)]))
    
    PBaseTH = parametersDict["TBase"][:3,-1]
    RBaseTH = R.from_matrix(parametersDict["TBase"][:3,:3])

    parametersDict["TTool"] = np.array(parametersDict["TTool"])
    
    parametersTH = np.hstack((PBaseTH,R.as_euler(RBaseTH,"xyz"),parametersDict["pi"],parametersDict["TTool"][:,:3,-1].flatten()))
    return(parametersTH)
