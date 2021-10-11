import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

import sys
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cmap = plt.cm.get_cmap('hsv', 100)

from random import randint
from copy import deepcopy

### USEFUL MATRICES AND FUNCTIONS

def size(L):
    if type(L) == list:
        return(sum(size(l) for l in L))
    else:
        return(1) 

def display(M):
    Mdisplay = pd.DataFrame(np.round(M,3))
    print(Mdisplay)

def plotRobot(q,pi,ax=None,color=None):
    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if(color is None):
        color = cmap(randint(1,100) - 1)

    T = np.eye(4)
    lastJointPosition = np.array([0,0,0])

    for i in range(len(q)):

        T = np.dot(T,Ti(i,q,pi)[0])
        currentJointPosition = T[:3,-1]

        ax.scatter(currentJointPosition[0],currentJointPosition[1],currentJointPosition[2],color=color)
        ax.plot([lastJointPosition[0],currentJointPosition[0]],[lastJointPosition[1],currentJointPosition[1]],[lastJointPosition[2],currentJointPosition[2]],color='k',linestyle='--')

        lastJointPosition = deepcopy(currentJointPosition)

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    plt.show()

HR = np.array([[[0,0,0,0],
                [0,0,-1,0],
                [0,1,0,0],
                [0,0,0,0]],
                
                [[0,0,1,0],
                [0,0,0,0],
                [-1,0,0,0],
                [0,0,0,0]],
                
                [[0,-1,0,0],
                [1,0,0,0],
                [0,0,0,0],
                [0,0,0,0]]])

HT = np.array([[[0,0,0,1],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]],
                
                [[0,0,0,0],
                [0,0,0,1],
                [0,0,0,0],
                [0,0,0,0]],
                
                [[0,0,0,0],
                [0,0,0,0],
                [0,0,0,1],
                [0,0,0,0]]])

### ROBOT GEOMETRIC MODEL (revolute joints only)

#Number of revolute joints
JOINTS_NUMBER = 7

#Joints axis index (x (0), y (1) or z (2))
JOINTS_AXIS = [2,1,2,1,2,1,2]

#Angular offset at each joint (yes (1) or no (0))
JOINTS_OFFSET = [0,1,1,1,1,1,0]

#Links translations axis (along x (0), y (1) or z (2))
LINKS_TRANSLATIONS = [[0,1],
                      [0,2],
                      [0,1],
                      [0,2],
                      [0,1],
                      [0,2],
                      []]

#Links rotations axis (around x (0), y (1) or z (2))
LINKS_ROTATIONS = [[0],
                   [0],
                   [0],
                   [0],
                   [0],
                   [0],
                   []]

#Indices of the first joint/link i parameter in parameters vector
PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER)])
PARAMETERS_INDEX = np.append(PARAMETERS_INDEX,[PARAMETERS_INDEX[-1]])

#Nominal parameters

d1 = 0.333
d2 = 0.316
d3 = 0.384
d4 = 0.088
d5 = -0.107

pi0 = np.array([0,0,0,      #1
                0,0,d2,0,   #2
                0,0,0,0,    #3
                0,0,d3,0,   #4
                0,0,0,0,    #5
                0,d4,0,0])  #6

#Nominal robot base transformation
Tbase = np.eye(4)
Tbase[2,-1] = d1

#Nominal robot tool transformation
Ttool = np.eye(4)
Ttool[:3,:3] = R.from_euler("xyz",[np.pi,0,0]).as_matrix()
Ttool[2,-1] = d5

def Ti(i,q,pi):

    splitT = []
    T = np.eye(4)

    parameters = pi[PARAMETERS_INDEX[i]:PARAMETERS_INDEX[i+1]]

    ###Joint rotation and offset
    Ti = np.eye(4)
    angles = np.zeros(3)

    angles[JOINTS_AXIS[i]] = q[i]
    Ti[:3,:3] = R.from_euler("xyz",angles).as_matrix()
    splitT.append(Ti)
    T = np.dot(T,Ti)

    if(JOINTS_OFFSET[i] == 1):
        Ti = np.eye(4)
        angles = np.zeros(3)

        angles[JOINTS_AXIS[i]] = parameters[0]
        Ti[:3,:3] = R.from_euler("xyz",angles).as_matrix()
        splitT.append(Ti)
        T = np.dot(T,Ti)

    ###Link translation
    for n,axis in enumerate(LINKS_TRANSLATIONS[i]):
        Ti = np.eye(4)
        Ti[axis,-1] = parameters[JOINTS_OFFSET[i] + n]

        splitT.append(Ti)
        T = np.dot(T,Ti)

    ###Link rotation
    for n,axis in enumerate(LINKS_ROTATIONS[i]):
        Ti = np.eye(4)
        angles = np.zeros(3)

        angles[axis] = parameters[JOINTS_OFFSET[i] + len(LINKS_TRANSLATIONS[i]) + n]
        Ti[:3,:3] = R.from_euler("xyz",angles).as_matrix()

        splitT.append(Ti)
        T = np.dot(T,Ti)

    return(T,splitT)

def T(q,pi):
    T = np.eye(4)
    for i in range(JOINTS_NUMBER):
        T = np.dot(T,Ti(i,q,pi)[0])
        #print(Ti(i,q,pi)[0])
    return(T)

#q0 = np.zeros(7)
#display(T(q0,pi0))

#q1 = np.zeros(7)
#q1[1] = np.pi/2
#display(T(q1,pi0))

#plotRobot(q1,pi0)

### ROBOT IDENTIFICATION JACOBIAN MATRIX

def dTk(k,q,pi):

    dTk = np.eye(4)
    i = np.argwhere(PARAMETERS_INDEX<=k)[-1][0] 
    #print("Concerned joint = " + str(i))

    for j in range(i):
        #print("Joint :" + str(j))
        dTk = np.dot(dTk,Ti(j,q,pi)[0])

    _,sTi = Ti(i,q,pi)
    
    for m in sTi[:k-PARAMETERS_INDEX[i]+1]:
        #print("Matrix : " + str(m))
        dTk = np.dot(dTk,m)

    if(k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] < 0):
        #print("offset")
        dTk = np.dot(dTk,HR[JOINTS_AXIS[i]])

    elif(k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] - size(LINKS_TRANSLATIONS[i]) < 0):
        #print("translation")
        #print("axis : "+str(k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] + 1))
        dTk = np.dot(dTk,HT[k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] + 1])
    
    else:
        #print("rotation")
        #print("axis : "+str(k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] - size(LINKS_TRANSLATIONS[i]) + 1))
        dTk = np.dot(dTk,HR[k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] - size(LINKS_TRANSLATIONS[i]) + 1])

    for m in sTi[k-PARAMETERS_INDEX[i]+2:]:
        #print("Matrix : " + str(m))
        dTk = np.dot(dTk,m)

    for j in range(i+1,JOINTS_NUMBER):
        #print("Joint :" + str(j))
        dTk = np.dot(dTk,Ti(j,q,pi)[0])

    return(dTk)

def Jpi(q,pi):
    Jpi = np.zeros((6,len(pi)))
    Tinverse = np.linalg.inv(T(q,pi))

    for k in range(len(pi)):
        Tk = dTk(k,q,pi)
        TkT = np.dot(Tk,Tinverse)

        Jpi[0,k] = Tk[0,3]
        Jpi[1,k] = Tk[1,3]
        Jpi[2,k] = Tk[2,3]

        Jpi[3,k] = TkT[2,1]
        Jpi[4,k] = TkT[0,2]
        Jpi[5,k] = TkT[1,0]

        display(Jpi)

    return(Jpi)