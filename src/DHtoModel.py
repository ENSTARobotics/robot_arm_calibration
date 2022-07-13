import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

###USEFULL FUNCTIONS

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

d1 = 0.1807
a2 = 0.6127
a3 = 0.57155 
d4 = 0.17415
d5 = 0.11985
d6 = 0.11655

#theta, a, d, alpha
DHParameters = np.array([[0,0,d1,-np.pi/2],
                         [0,-a2,0,0],
                         [0,-a3,0,0],
                         [0,0,d4,-np.pi/2],
                         [0,0,d5,np.pi/2],
                         [0,0,d6,0]])

initialJointAxis = np.array([0,0,1])

R = lambda alpha : np.array([[1,0,0],
                             [0,np.cos(alpha),-np.sin(alpha)],
                             [0,np.sin(alpha),np.cos(alpha)]])

PI = []
defaultT = ["Tx","Ty","Tz"]
defaultR = ["Rx","Ry","Rz"]

jointsAxis = []
jointsOffsets = []

linksTranslations = []
linksRotations = []

pi0 = []

currentJointAxis = deepcopy(initialJointAxis)
currentJointAxisIndex = np.where(np.abs(currentJointAxis) == 1)[0][0]
currentJointAxisID = int((currentJointAxisIndex + 1)*np.sign(currentJointAxis[currentJointAxisIndex]))

additionnalDistancesCounter = [0,0,0]

for i,parameters in enumerate(DHParameters):

    jointsAxis.append(currentJointAxisID)

    if(i == len(DHParameters) - 1):
        jointsOffsets.append(0)
        linksRotations.append([])
        linksTranslations.append([])
        break
  
    #Joint rotation and offset
    if(i != 0):
        jointsOffsets.append(0)
        PI.append("deltaQ"+str(i+1))
        pi0.append(0)

    #Compute new joint axis
    newJointAxis = np.dot(R(parameters[-1]),currentJointAxis)
    try:
        newJointAxisIndex = np.where(np.abs(newJointAxis) == 1)[0][0]
        newJointAxisID = int((newJointAxisIndex + 1)*np.sign(newJointAxis[newJointAxisIndex]))
    except IndexError:
        raise NotImplementedError

    #If ej-1 _|_ ej, we delete Ru/v,j-1 which corresponds to Re,j
    if(newJointAxisIndex != currentJointAxisIndex):
        if(i == len(DHParameters) - 2):
            currentT = [item for i,item in enumerate(defaultT) if i != currentJointAxisIndex and i != newJointAxisIndex]
        else:
            currentT = [item for i,item in enumerate(defaultT) if i != currentJointAxisIndex]

        currentR = [item for i,item in enumerate(defaultR) if i != currentJointAxisIndex and i != newJointAxisIndex]

    #If ej-1 // ej, we delete Tu/v,j-k which defines the translation orthogonal to ej, with k minimal
    else:
        if(i == len(DHParameters) - 2):
            currentT = [item for i,item in enumerate(defaultT) if i != currentJointAxisIndex and i != newJointAxisIndex]
        else:
            currentT = [item for i,item in enumerate(defaultT) if i != currentJointAxisIndex and i != max((currentJointAxisIndex-1)%3,(currentJointAxisIndex-2)%3)]
        currentR = [item for i,item in enumerate(defaultR) if i != currentJointAxisIndex]

    linksTranslations.append([i + 1 for i,item in enumerate(defaultT) if item in currentT])
    linksRotations.append([i + 1 for i,item in enumerate(defaultR) if item in currentR])

    nextParameters = DHParameters[i+1]
    commonNormal = (max(newJointAxisIndex,currentJointAxisIndex) + np.abs(newJointAxisIndex-currentJointAxisIndex))%3

    if(nextParameters[1] != 0 and defaultT[commonNormal] in currentT):
        

    if(nextParameters[2] != 0):
        additionnalDistancesCounter[newJointAxisIndex] += nextParameters[2]

    PI.extend(currentT)
    PI.extend(currentR)

    currentJointAxis = deepcopy(newJointAxis)
    currentJointAxisID = newJointAxisID
    currentJointAxisIndex = newJointAxisIndex

