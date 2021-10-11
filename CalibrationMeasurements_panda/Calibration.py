#!/usr/bin/python3

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.defchararray import array, join
from scipy.optimize import leastsq, zeros, least_squares

from scipy.spatial.transform import Rotation as R, rotation

from random import randint

from copy import deepcopy

cmap = plt.cm.get_cmap('hsv', 100)

rowCounter = 0

optiTrackPose = []
encodersPose = []

robotBaseUR = []
robotBaseUL = []
robotBaseDR = []
robotBaseDL = []

motorsAngles = []
jointsAngles = []
desiredJointsAngles = []

efforts = []

motorTorques = []
desiredMotorTorques = []
externalTroques = []

def csvToArray(csvRow):
    arrayRow = []
    for csvItem in csvRow:
        csvItem.replace(',','.')
        arrayItem = float(csvItem)
        arrayRow.append(arrayItem)
    return(arrayRow)

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

def drawPose(pose, ax = None, color = None):
    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if(color is None):
        color = cmap(randint(1,100) - 1)

    ax.scatter(pose[0],pose[1],pose[2],color=color)

    eulerAngles = R.from_euler("xyz",[pose[3],pose[4],pose[5]])
    rotationMatrix = eulerAngles.as_matrix()

    ax.quiver(pose[0],pose[1],pose[2],rotationMatrix[0,0]/10,rotationMatrix[1,0]/10,rotationMatrix[2,0]/10,color='r')
    ax.quiver(pose[0],pose[1],pose[2],rotationMatrix[0,1]/10,rotationMatrix[1,1]/10,rotationMatrix[2,1]/10,color='g')
    ax.quiver(pose[0],pose[1],pose[2],rotationMatrix[0,2]/10,rotationMatrix[1,2]/10,rotationMatrix[2,2]/10,color='b')

with open("/home/caroline/Bureau/CalibrationMeasurements_panda/Calibration.csv", newline='') as csvfile:
    rowReader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in rowReader:
        if(row[0]=="---"):
            rowCounter = 0
        else:
            if(rowCounter == 0):
                optiTrackPose.append(csvToArray(row))

            elif(rowCounter == 1):
                robotBaseUR.append(csvToArray(row))

            elif(rowCounter == 2):
                robotBaseUL.append(csvToArray(row))

            elif(rowCounter == 3):
                robotBaseDR.append(csvToArray(row))

            elif(rowCounter == 4):
                robotBaseDL.append(csvToArray(row))

            elif(rowCounter == 5):
                encodersPose.append(csvToArray(row))

            elif(rowCounter == 6):
                motorsAngles.append(csvToArray(row))

            elif(rowCounter == 7):
                jointsAngles.append(csvToArray(row))

            elif(rowCounter == 8):
                desiredJointsAngles.append(csvToArray(row))

            elif(rowCounter == 9):
                efforts.append(csvToArray(row))
            
            elif(rowCounter == 10):
                motorTorques.append(csvToArray(row))

            elif(rowCounter == 11):
                desiredMotorTorques.append(csvToArray(row))

            elif(rowCounter == 12):
                externalTroques.append(csvToArray(row))
            
            rowCounter += 1

optiTrackPose = np.array(optiTrackPose)
encodersPose = np.array(encodersPose)

robotBaseUR = np.array(robotBaseUR)
robotBaseUL = np.array(robotBaseUL)
robotBaseDR = np.array(robotBaseDR)
robotBaseDL = np.array(robotBaseDL)

motorsAngles = np.array(motorsAngles)
jointsAngles = np.array(jointsAngles)
desiredJointsAngles = np.array(desiredJointsAngles)

efforts = np.array(efforts)

motorTorques = np.array(motorTorques)
desiredMotorTorques = np.array(desiredMotorTorques)
externalTroques = np.array(externalTroques)

def planeInterpolation(points):
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

def planeProjection(normal,vector):
    return(vector - (np.dot(vector,normal)/np.linalg.norm(normal)**2)*normal)

robotBase = np.array([filteredMean(robotBaseUR,axis=0),filteredMean(robotBaseUL,axis=0),filteredMean(robotBaseDR,axis=0),filteredMean(robotBaseDL,axis=0)])

P = planeInterpolation(robotBase[:,:3].T)

z = P[:3]
robotBase[:,:3] -= (0.016 + 0.00635 - 0.00075)*z
y = planeProjection(z,(robotBase[1]-robotBase[0]+robotBase[3]-robotBase[2])[:3])
y /= np.linalg.norm(y)

x = np.cross(y,z)

robotBaseRotation = R.from_matrix(np.array([x,y,z]))
robotBasePosition = (robotBase[1][:3]+robotBase[0][:3])/2 + 0.011*x

robotBaseTransformation = np.concatenate((np.concatenate((np.array([x,y,z]),np.array([robotBasePosition]).T),axis=1),np.array([[0,0,0,1]])))

print(robotBaseRotation.as_euler("xyz"))
print(robotBasePosition)

Delta = np.empty((0,1))

markersPose = []

DeltaMatrix = []
localDeltaMatrix = []
currentIndex = 0

for i,position in enumerate(deepcopy(encodersPose)):

    encodersRotation = R.from_euler("xyz",[position[3],position[4],position[5]])

    position[:3] += encodersRotation.apply(np.array([0.0349 + 0.010,- 0.0034 - 0.0125, 0])) #Delta 0.05047
    position[:3] = robotBaseRotation.apply(position[:3]) + robotBasePosition

    rotationMatrix = robotBaseRotation*encodersRotation
    position[3:] = rotationMatrix.as_euler("xyz")

    markersPose.append(position)

    markerRotation = R.from_euler("xyz",optiTrackPose[i][3:])*R.from_euler("xyz",[0,np.pi,0])
    optiTrackPose[i][3:] = markerRotation.as_euler("xyz")

    currentDelta = np.reshape(optiTrackPose[i] - position,(len(position),1))
    currentDelta[3:] = modAngle(currentDelta[3:])
    
    Delta = np.concatenate((Delta,currentDelta))

    if(i == 0):
        continue

    elif(i == 1):
        localDeltaMatrix.append(currentDelta)

    else:
        index = np.where(np.round(np.abs(jointsAngles[i] - jointsAngles[i-1]),3) != 0)[0][-1]
        if(index != currentIndex):
            DeltaMatrix.append(deepcopy(localDeltaMatrix))
            localDeltaMatrix = [currentDelta]
            currentIndex = index
        else:
            localDeltaMatrix.append(currentDelta)

DeltaMatrix.append(deepcopy(localDeltaMatrix))

markersPose = np.array(markersPose)
L = 0

def sineInterpolation(X,Y):
    def delta(parameters):
        f = lambda x : parameters[0]*np.sin(parameters[1]*x+parameters[2]) + parameters[3]
        return(np.abs(Y.flatten() - f(X)))

    initialGuess = [np.abs(max(Y)),1,0,np.average(Y)]

    interpolation = least_squares(delta,initialGuess)["x"]

    f = lambda x : interpolation[0]*np.sin(interpolation[1]*x+interpolation[2]) + interpolation[3]
    return(interpolation,f)

X = jointsAngles[1+len(DeltaMatrix[0])+len(DeltaMatrix[1])+len(DeltaMatrix[2])+len(DeltaMatrix[3])+len(DeltaMatrix[4])+len(DeltaMatrix[5]):1+len(DeltaMatrix[0])+len(DeltaMatrix[1])+len(DeltaMatrix[2])+len(DeltaMatrix[3])+len(DeltaMatrix[4])+len(DeltaMatrix[5])+len(DeltaMatrix[0])+len(DeltaMatrix[1])+len(DeltaMatrix[2])+len(DeltaMatrix[3])+len(DeltaMatrix[4])+len(DeltaMatrix[6]),6]
Y1 = np.array(DeltaMatrix[6])[:,0]
Y2 = np.array(DeltaMatrix[6])[:,1]
Y3 = np.array(DeltaMatrix[6])[:,2]

P1,f1 = sineInterpolation(X,Y1)
P2,f2 = sineInterpolation(X,Y2)
P3,f3 = sineInterpolation(X,Y3)

print(P1)
print(P2)
print(P3)

plt.plot(X,Y1,'b',label="X")
plt.plot(X,f1(X),'b--')
plt.plot(X,Y2,'r',label="Y")
plt.plot(X,f2(X),'r--')
plt.plot(X,Y3,'k',label="Z")
plt.plot(X,f3(X),'k--')
plt.legend()
plt.show()

plt.figure()

for i,item in enumerate(DeltaMatrix):
    currentItem = np.array(item)
    ax = plt.subplot(2,4,1+i)
    ax.set_title("Axis " + str(i+1))
    #ax.set_xlabel("Joint value (in rad)")
    #ax.set_ylabel(r"$\Delta$ (in m)")
    ax.plot(jointsAngles[1+L:1+len(item)+L,i],currentItem[:,0],label=r"$\Delta$ X")
    ax.plot(jointsAngles[1+L:1+len(item)+L,i],currentItem[:,1],label=r"$\Delta$ Y")
    ax.plot(jointsAngles[1+L:1+len(item)+L,i],currentItem[:,2],label=r"$\Delta$ Z")
    L+=len(item)

plt.legend()
plt.show()

"""
for i in range(1,10):
    print(markersPose[i])
    print(optiTrackPose[i])
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')

    for point in robotBase:
        ax.scatter(point[0],point[1],point[2])

    ax.scatter(robotBasePosition[0],robotBasePosition[1],robotBasePosition[2])
    ax.quiver(robotBasePosition[0],robotBasePosition[1],robotBasePosition[2],x[0]/10,x[1]/10,x[2]/10)
    ax.quiver(robotBasePosition[0],robotBasePosition[1],robotBasePosition[2],y[0]/10,y[1]/10,y[2]/10)
    ax.quiver(robotBasePosition[0],robotBasePosition[1],robotBasePosition[2],z[0]/10,z[1]/10,z[2]/10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    drawPose(markersPose[i],ax,color='b')
    drawPose(optiTrackPose[i],ax,color='r')

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    plt.show()
"""

masses =  np.array([4.971, 0.647, 3.229, 3.588, 1.226, 1.667, 0.736, 0.046])
jointsNumber = len(masses)

               #xG, yG, zG
centersOfMass = np.array([[0.003875, 0.002081, 0.0],
                [-0.003141, -0.02872, 0.003495],
                [0.027518, 0.039252, -0.066502],
                [-0.05317, 0.104419, 0.027454],
                [-0.011953, 0.041065, -0.038437],
                [0.060149, -0.014117, -0.010517],
                [0.010517, -0.004252, 0.061597],
                [0.00165, -0.00071, 0.00601]])

              #a, alpha, d
KKParameters = np.array([[0, 0, 0.333, 0],
               [0, -1.570796, 0, 0],
               [0, 1.570796, 0.316, 0],
               [0.0825, 1.570796, 0, 0],
               [-0.0825, -1.570796, 0.384, 0],
               [0, 1.570796, 0, 0],
               [0.088, 1.570796, 0, 0],
               [0, 0, 0.1182, -0.7854]])

def compute_Tij(i,j,theta):
    Tij = np.eye(4)

    if(i <= j):
        tmpT = np.eye(4)

        for k in range(j-1,i-1,-1):
            tmpT[0,0] = np.cos(theta[k]+KKParameters[k][3])
            tmpT[0,1] = -np.sin(theta[k]+KKParameters[k][3])
            tmpT[0,3] = KKParameters[k][0]

            tmpT[1,0] = np.sin(theta[k]+KKParameters[k][3])*np.cos(KKParameters[k][1])
            tmpT[1,1] = np.cos(theta[k]+KKParameters[k][3])*np.cos(KKParameters[k][1])
            tmpT[1,2] = -np.sin(KKParameters[k][1])
            tmpT[1,3] = -KKParameters[k][2]*np.sin(KKParameters[k][1])

            tmpT[2,0] = np.sin(theta[k]+KKParameters[k][3])*np.sin(KKParameters[k][1])
            tmpT[2,1] = np.cos(theta[k]+KKParameters[k][3])*np.sin(KKParameters[k][1])
            tmpT[2,2] = np.cos(KKParameters[k][1])
            tmpT[2,3] = KKParameters[k][2]*np.cos(KKParameters[k][1])

            Tij = np.dot(tmpT,Tij)
        
        return(Tij)

    else:
        return(np.linalg.inv(compute_Tij(j,i)))

def drawRobot(jointsValues, ax = None, color = None):

    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if(color is None):
        color = cmap(randint(1,100) - 1)

    lastJointPosition = np.array([0,0,0])

    for i in range(jointsNumber+1):
        T0i = compute_Tij(0,i,jointsValues)
        currentJointPosition = T0i[:-1,-1]

        ax.scatter(currentJointPosition[0],currentJointPosition[1],currentJointPosition[2],color=color)
        ax.plot([lastJointPosition[0],currentJointPosition[0]],[lastJointPosition[1],currentJointPosition[1]],[lastJointPosition[2],currentJointPosition[2]],color='k',linestyle='--')

        lastJointPosition = deepcopy(currentJointPosition)

def compute_Wi(i,theta):
    Wi = np.zeros((6,6))

    Ti = compute_Tij(0,i,theta)

    Wi[:3,:3] = Ti[:3,:3]
    Wi[3:,3:] = Ti[:3,:3]

    tmpr = (compute_Tij(0,i,theta) - compute_Tij(0,jointsNumber,theta))[:3,-1]
    
    Wi[:3,3:] = np.dot(np.array([[0,-tmpr[2],tmpr[1]],[tmpr[2],0,-tmpr[0]],[-tmpr[1],tmpr[0],0]]),Ti[:3,:3])

    return(Wi)

def compute_fGi(i,theta):
    fGi = np.zeros((6,1))

    fGi[2] = 1

    tmpr = (compute_Tij(0,i,theta) - compute_Tij(0,jointsNumber,theta))[:3,-1]
    tmpV = tmpr + np.dot(compute_Tij(0,i,theta)[:3,:3],centersOfMass[i-1])
    tmpV = np.cross(tmpV,np.array([0,0,1]))

    fGi[3:] = np.reshape(tmpV[:3],(3,1))

    fGi *= -masses[i-1]*10

    return(fGi)

def compute_WMatrix(theta):
    wMatrix = np.zeros((6*jointsNumber,6*jointsNumber))

    for i in range(jointsNumber):
        wMatrix[6*i:6*i+6,6*i:6*i+6] = compute_Wi(i+1,theta)

    return(wMatrix)

def compute_ComplianceMatrix(compliances):
    complianceMatrix = np.zeros((6*jointsNumber,6*jointsNumber))
    
    for i in range(jointsNumber):
        complianceMatrix[6*i+5,6*i+5] = compliances[i]

    return(complianceMatrix)

def compute_forceVector(theta):
    forceVector = np.zeros((6*jointsNumber,1))
    blockForceVector = np.zeros((6,1))

    for i in range(jointsNumber,0,-1):
        blockForceVector += compute_fGi(i,theta)
        forceVector[6*i-6:6*i] = blockForceVector
    
    return(forceVector)

def delta(compliances,theta):
    delta = np.empty((0,1))
    M = np.eye(6)
    M = np.hstack([M]*jointsNumber)

    W = compute_WMatrix(theta)
    
    C = compute_ComplianceMatrix(compliances)

    fG = compute_forceVector(theta)

    delta = np.dot(np.dot(M,np.dot(np.dot(W,C),W.T)),fG)

    return(delta)

def function(compliances):

    computedDelta = np.empty((0,1))

    for joints in jointsAngles:
        theta = np.concatenate((joints,np.zeros(1)))
        computedDelta = np.concatenate((computedDelta,delta(compliances,theta)))

    result = np.array((computedDelta-Delta)**2, dtype=float)*1000
    result = result.flatten()

    return(np.array(result,dtype=float))
    
#compliances = np.ones(jointsNumber)
#compliances[0] = 0
#compliances[-1] = 0
#compliances = np.array(compliances,dtype=float)

#result = least_squares(function,compliances)
#C = result['x']
#print(C)

"""
DTheta = []

for index in range(len(encodersPose)):
    print(index)

    def function2(theta):

        theta = np.append(theta,0)
        T0E = compute_Tij(0,jointsNumber,theta)
        encodersRotation = R.from_matrix(T0E[:3,:3])
        position[:3] = T0E[:3,-1]
        position[3:] = encodersRotation.as_euler("xyz")

        position[:3] += encodersRotation.apply(np.array([0.0349 + 0.010,- 0.0034 - 0.0125,0]))
        position[:3] = robotBaseRotation.apply(position[:3]) + robotBasePosition

        rotationMatrix = robotBaseRotation*encodersRotation
        position[3:] = rotationMatrix.as_euler("xyz")

        currentDelta = np.reshape(optiTrackPose[index] - position,(len(position),1))
        currentDelta[3:] = modAngle(currentDelta[3:])
        currentDelta = currentDelta.flatten()

        return(currentDelta)

    result = least_squares(function2,jointsAngles[index])
    T = result['x']

    DTheta.append(jointsAngles[index]-T)

DTheta = np.array(DTheta)

for i in range(jointsNumber - 1):
    plt.plot(np.arange(0,len(encodersPose),1),DTheta[:,i],'--')
    plt.plot(np.arange(0,len(encodersPose),1),jointsAngles[:,i])
    plt.legend()
    plt.show()

for i in range(10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(encodersPose[i])
    print(compute_Tij(0,jointsNumber,np.append(jointsAngles[i],0)))

    drawPose(encodersPose[i],ax=ax,color="black")
    drawRobot(np.append(jointsAngles[i],0),ax=ax,color="blue")

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    plt.show()
"""