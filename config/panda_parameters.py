#!/usr/bin/python3

import numpy as np
import os

robotName = os.path.basename(__file__).split("_")[0]

d1 = 0.333
d2 = 0.316
d3 = 0.0825
d4 = 0.384
d5 = 0.088
d6 = 0.107

#Per line : alpha, d, r, theta - Modified parameters convention !
DHParameters = np.array([[0,0,d1,0],
                         [-np.pi/2,0,0,0],
                         [np.pi/2,0,d2,0],
                         [np.pi/2,d3,0,0],
                         [-np.pi/2,-d3,d4,0],
                         [np.pi/2,0,0,0],
                         [np.pi/2,d5,d6,0]])

if(not os.path.exists(os.path.dirname(os.path.realpath(__file__)) + "/" + robotName + "_parameters.npy")):
    from DHtoParameters import DHtoParameters
    DHtoParameters(DHParameters,robotName)

with open(os.path.dirname(os.path.realpath(__file__)) + "/" + robotName + "_parameters.npy","rb") as file:
    JOINTS_NUMBER = np.load(file, allow_pickle=True)
    JOINTS_AXIS = np.load(file, allow_pickle=True)
    JOINTS_OFFSET = np.load(file, allow_pickle=True)
    LINKS_TRANSLATIONS = np.load(file, allow_pickle=True)
    LINKS_ROTATIONS = np.load(file, allow_pickle=True)
    pi0 = np.load(file, allow_pickle=True)
    TBaseN = np.load(file, allow_pickle=True)
    TFlangeN = np.load(file, allow_pickle=True)

robotBaseOffset = np.array([[0,-1,0,0],
                            [1,0,0,(0.120/2) - 0.020 + 0.011],
                            [0,0,1,0.00565],
                            [0,0,0,1]])
robotBaseItemsNames = ["robotBase1","robotBase2","robotBase3","robotBase4","robotBase5","robotBase6","robotBase7"]

def planeProjection(point,planeNormal,planePoint):
    v = point - planePoint
    dist = np.dot(planeNormal,v)
    return(point - dist*planeNormal)

def robotBaseTransformation(Items):
    
    # ____________________________
    #|(6)                      (1)|
    #|                            |  
    #|                      (7)   |
    #|                            |
    #|             _|_       (2)  |  
    #|  (5)         |             |      Y
    #|                            |      |
    #|                            |     -|-- X
    #|                            |
    #|(4)______________________(3)|          
    #           
        
    #Retrieve the robot base markers positions
    robotBase = np.array([np.array([np.mean([coordinate for coordinate in array if np.abs(coordinate - np.mean(array)) <= np.std(array)]) for array in Items[item].T]) for item in robotBaseItemsNames])
    baseCentroid = np.mean(robotBase[:-1],axis=0)

    #import matplotlib.pyplot as plt
    #ax = plt.figure().add_subplot(projection='3d')
    #scatter = ax.scatter(robotBase[:,0],robotBase[:,1],robotBase[:,2],c=np.arange(len(robotBase)))
    #ax.scatter(baseCentroid[0],baseCentroid[1],baseCentroid[2])
    #extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    #sz = extents[:,1] - extents[:,0]
    #centers = np.mean(extents, axis=1)
    #maxsize = max(abs(sz))
    #r = maxsize/2
    #for ctr, dim in zip(centers, 'xyz'):
    #    getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
    #ax.set_xlabel("x")
    #ax.set_ylabel("y")
    #ax.set_zlabel("z")
    #plt.colorbar(scatter)
    #plt.show()

    #Find the plane normal
    z = np.dot(np.linalg.pinv(robotBase),np.ones(len(robotBase)))
    z/=np.linalg.norm(z)
    if(z[2] < 0):
        z*=-1

    #Find the other axis
    #x = (robotBase[0] - robotBase[5]) #OLD WAY (WRONG, BUT LEGACY)
    x = (planeProjection(robotBase[0],z,baseCentroid) - planeProjection(robotBase[5],z,baseCentroid))  #CURRENT LAYOUT FOR PANDA
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