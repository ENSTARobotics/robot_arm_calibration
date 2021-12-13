import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import disp

from scipy.spatial.transform import Rotation as R

import sys

np.set_printoptions(threshold=sys.maxsize)

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cmap = plt.cm.get_cmap('hsv',100)

from random import randint
from copy import deepcopy

from importlib import import_module
from sys import argv

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(argv[1]+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    #print("INVALID ROBOT NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

### USEFUL MATRICES AND FUNCTIONS

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

def size(L):
    if type(L) == list:
        return(sum(size(l) for l in L))
    else:
        return(1) 

def display(M):
    Mdisplay = pd.DataFrame(np.round(M,3))
    print(Mdisplay)

def plotRobot(q,pi,TBase=np.eye(4),TTool=np.eye(4),ax=None,color=None):
    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if(color is None):
        color = cmap(randint(1,100) - 1)

    T = np.eye(4)
    lastJointPosition = np.array([0,0,0])

    T = np.dot(T,TBase)
    currentJointPosition = T[:3,-1]

    ax.scatter(currentJointPosition[0],currentJointPosition[1],currentJointPosition[2],color=color)

    lastJointPosition = deepcopy(currentJointPosition)

    for i,_ in enumerate(q):

        T = np.dot(T,Ti(i,q,pi)[0])
        currentJointPosition = T[:3,-1]

        ax.scatter(currentJointPosition[0],currentJointPosition[1],currentJointPosition[2],color=color)
        ax.plot([lastJointPosition[0],currentJointPosition[0]],[lastJointPosition[1],currentJointPosition[1]],[lastJointPosition[2],currentJointPosition[2]],color='k',linestyle='--')

        lastJointPosition = deepcopy(currentJointPosition)

    T = np.dot(T,TTool)
    currentJointPosition = T[:3,-1]

    ax.scatter(currentJointPosition[0],currentJointPosition[1],currentJointPosition[2],color=color)
    ax.plot([lastJointPosition[0],currentJointPosition[0]],[lastJointPosition[1],currentJointPosition[1]],[lastJointPosition[2],currentJointPosition[2]],color='k',linestyle='--')

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

### ROBOT GEOMETRIC MODEL (revolute joints only)

#Indices of the first joint/link i parameter in parameters vector
PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER+1)])

def Ti(i,q,pi):

    splitT = []
    T = np.eye(4)

    parameters = pi[PARAMETERS_INDEX[i]:PARAMETERS_INDEX[i+1]]

    ###Joint rotation and offset
    Ti = np.eye(4)
    angles = np.zeros(3)

    angles[np.abs(JOINTS_AXIS[i])-1] = np.sign(JOINTS_AXIS[i])*q[i]
    Ti[:3,:3] = R.from_euler("xyz",angles).as_matrix()
    splitT.append(Ti)
    T = np.dot(T,Ti)

    if(JOINTS_OFFSET[i] == 1):
        Ti = np.eye(4)
        angles = np.zeros(3)

        angles[np.abs(JOINTS_AXIS[i])-1] = parameters[0]
        Ti[:3,:3] = R.from_euler("xyz",angles).as_matrix()
        splitT.append(Ti)
        T = np.dot(T,Ti)

    ###Link translation
    for n,axis in enumerate(LINKS_TRANSLATIONS[i]):
        Ti = np.eye(4)
        Ti[np.abs(axis)-1,-1] = np.sign(axis)*parameters[JOINTS_OFFSET[i] + n]

        splitT.append(Ti)
        T = np.dot(T,Ti)

    ###Link rotation
    for n,axis in enumerate(LINKS_ROTATIONS[i]):
        Ti = np.eye(4)
        angles = np.zeros(3)

        angles[np.abs(axis)-1] = np.sign(axis)*parameters[JOINTS_OFFSET[i] + len(LINKS_TRANSLATIONS[i]) + n]
        Ti[:3,:3] = R.from_euler("xyz",angles).as_matrix()

        splitT.append(Ti)
        T = np.dot(T,Ti)

    return(T,splitT)

def T(q,pi,TBase=np.eye(4),TTool=np.eye(4)):
    T = np.eye(4)
    for i in range(JOINTS_NUMBER):
        T = np.dot(T,Ti(i,q,pi)[0])
    return(np.dot(np.dot(TBase,T),TTool))

### ROBOT IDENTIFICATION JACOBIAN MATRIX

def dTk(k,q,pi,TBase=np.eye(4),TTool=np.eye(4)):

    dTk = np.eye(4)
    i = np.argwhere(PARAMETERS_INDEX<=k)[-1][0] 
    #print("Concerned joint = " + str(i+1))

    for j in range(i):
        #print("Previous joint :" + str(j+1))
        dTk = np.dot(dTk,Ti(j,q,pi)[0])

    _,sTi = Ti(i,q,pi)
    
    for m in sTi[:k-PARAMETERS_INDEX[i]+1]:
        #print("Previous transformation matrix :\n" + str(m))
        dTk = np.dot(dTk,m)

    if(k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] < 0):
        #print("Joint offset :\n" + str(HR[np.abs(JOINTS_AXIS[i])-1]))
        #print("axis : "+str(np.abs(JOINTS_AXIS[i])))
        dTk = np.dot(dTk,HR[np.abs(JOINTS_AXIS[i])-1])

    elif(k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] - size(LINKS_TRANSLATIONS[i]) < 0):
        #print("Link translation :\n" + str(HT[k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i]]))
        #print("axis : "+str(k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] + 1))
        dTk = np.dot(dTk,HT[k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i]])
    
    else:
        #print("Link rotation :\n" + str(HR[k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] - size(LINKS_TRANSLATIONS[i])]))
        #print("axis : "+str(k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] - size(LINKS_TRANSLATIONS[i]) + 1))
        dTk = np.dot(dTk,HR[k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] - size(LINKS_TRANSLATIONS[i])])

    for m in sTi[k-PARAMETERS_INDEX[i]+2:]:
        #print("Following transformation matrix :\n" + str(m))
        dTk = np.dot(dTk,m)

    for j in range(i+1,JOINTS_NUMBER):
        #print("Following joint : " + str(j+1))
        dTk = np.dot(dTk,Ti(j,q,pi)[0])

    return(np.dot(np.dot(TBase,dTk),TTool))

def Jpi(q,pi,TBase=np.eye(4),TTool=np.eye(4)):
    Jpi = np.empty((6,len(pi)))
    Tinverse = np.linalg.inv(T(q,pi,TBase,TTool))

    for k,_ in enumerate(pi):
        Tk = dTk(k,q,pi,TBase,TTool)
        TkT = np.dot(Tk,Tinverse)

        Jpi[0,k] = Tk[0,3]
        Jpi[1,k] = Tk[1,3]
        Jpi[2,k] = Tk[2,3]

        Jpi[3,k] = TkT[2,1]
        Jpi[4,k] = TkT[0,2]
        Jpi[5,k] = TkT[1,0]

    return(Jpi)

def JpiT(q,pi,TBase=np.eye(4),TTool=np.eye(4)):
    JpiT = np.empty((len(pi),6))
    Tinverse = np.linalg.inv(T(q,pi,TBase,TTool))

    for k,_ in enumerate(pi):
        Tk = dTk(k,q,pi,TBase,TTool)
        TkT = np.dot(Tk,Tinverse)

        JpiT[k,0] = Tk[0,3]
        JpiT[k,1] = Tk[1,3]
        JpiT[k,2] = Tk[2,3]

        JpiT[k,3] = TkT[2,1]
        JpiT[k,4] = TkT[0,2]
        JpiT[k,5] = TkT[1,0]

    return(JpiT)

### DEBUG

#q0 = np.zeros(JOINTS_NUMBER)
#display(T(q0,pi0,TBaseN,TToolN))
#plotRobot(q0,pi0,TBaseN,TToolN)
#
#q1 = np.zeros(JOINTS_NUMBER)
#q1[0] = np.pi/2
#display(T(q1,pi0,TBaseN,TToolN))
#plotRobot(q1,pi0,TBaseN,TToolN)

### PERFORMANCES

#import time
#from random import sample

#q = np.zeros(JOINTS_NUMBER)
#q[0] = np.pi/4
#q[1] = np.pi/2
#q[4] = np.pi/2
#q[6] = np.pi

#tic = time.perf_counter()
#Ti(0,q,pi0)
#toc = time.perf_counter()
#print("Single transformation computation : " + str(toc - tic) + " s")

#tic = time.perf_counter()
#T(q,pi0)
#toc = time.perf_counter()
#print("Complete transformation computation : " + str(toc - tic) + " s")

#tic = time.perf_counter()
#Jpi(q,pi0,np.eye(4),np.eye(4))
#toc = time.perf_counter()
#print("Identification Jacobian matrix computation : " + str(toc - tic) + " s")

