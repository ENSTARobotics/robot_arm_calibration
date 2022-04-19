#!/usr/bin/python3

import csv
import re
import numpy as np
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R

from collections import OrderedDict

from copy import deepcopy
import matplotlib.pyplot as plt

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

def robotBaseTransformation(Items,robotBaseMarkersOffset=0,robotBasePlaneOffset=np.zeros(3),robotBaseItemsNames=["robotBaseUR","robotBaseUL","robotBaseDR","robotBaseDL"]):

    ### Least-squares method for plane interpolation
    def planeInterpolation(points):
        #P : a*x + b*y + c*z + d = 0
        P0 = [0.0,0.0,1.0,0.0]

        def f_min(X,p):
            plane_xyz = p[0:3]
            distance = (plane_xyz*X.T).sum(axis=1) + p[3]
            return distance / np.linalg.norm(plane_xyz)

        def residuals(params, signal, X):
            return f_min(X, params)

        P = leastsq(residuals, P0, args=(None, points))[0]
        P /= np.linalg.norm(P[:3])

        return(P)

    ### Vector on plane projection
    def planeProjection(normal,vector):
        return(vector - (np.dot(vector,normal)/np.linalg.norm(normal)**2)*normal)

    #Retrieve the robot base markers positions
    robotBase = np.array([filteredMean(Items[robotBaseItemsNames[0]],axis=0),
                        filteredMean(Items[robotBaseItemsNames[1]],axis=0),
                        filteredMean(Items[robotBaseItemsNames[2]],axis=0),
                        filteredMean(Items[robotBaseItemsNames[3]],axis=0)])

    #Interpolate a plane passing through the 4 robot base marker points
    P = planeInterpolation(robotBase[:,:3].T)

    #Remove the screws vertical offset
    z = P[:3]
    robotBase[:,:3] -= robotBaseMarkersOffset*z

    #Compute the robot base frame x,y,z
    y = planeProjection(z,(robotBase[1]-robotBase[0]+robotBase[3]-robotBase[2])[:3])
    y /= np.linalg.norm(y)
    x = np.cross(y,z)

    #Compute the robot base position, based on the average of all robot base markers positions + an eventual user defined offset
    robotBasePosition = np.average(robotBase[:,:3],axis=0) + np.dot(np.array([x,y,z]),robotBasePlaneOffset)

    robotBaseTransformation = np.concatenate((np.concatenate((np.array([x,y,z]),np.array([robotBasePosition]).T),axis=1),np.array([[0,0,0,1]])))

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #for i,point in enumerate(robotBase):
    #    ax.scatter(point[0],point[1],point[2],label=robotBaseItemsNames[i])

    #ax.scatter(robotBasePosition[0],robotBasePosition[1],robotBasePosition[2])

    #extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    #sz = extents[:,1] - extents[:,0]
    #centers = np.mean(extents, axis=1)
    #maxsize = max(abs(sz))
    #r = maxsize/2
    #for ctr, dim in zip(centers, 'xyz'):
    #    getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    #ax.set_xlabel("x (m)")
    #ax.set_ylabel("y (m)")
    #ax.set_zlabel("z (m)")

    #plt.legend()
    #plt.show()
    
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
                if(np.shape(Items[itemsNames[rowCounter]]) == (0,0)):
                    Items[itemsNames[rowCounter]] = np.array([csvRowToArray(row)])
                else:
                    Items[itemsNames[rowCounter]] = np.append(Items[itemsNames[rowCounter]],np.array([csvRowToArray(row)]),axis=0)
                rowCounter+=1

    return(Items)


    

    