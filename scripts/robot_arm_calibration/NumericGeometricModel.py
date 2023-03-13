#!/usr/bin/python3

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

### Import robot and marker parameters
from robot_arm_calibration.ParametersTools import importParameters
importParameters()

### Useful matrices and functions

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
    """! Computes the size of a nested list as the sum of the sizes of all contained lists

    @param L   The nested list.

    @return  The size of the nested list.
    """
    if type(L) == list or type(L) == np.ndarray:
        return(sum(size(l) for l in L))
    else:
        return(1) 

### Robot geometric model (revolute joints only)

#Indices of the first joint/link i parameter in parameters vector
PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER+1)])

def Ti(i,q,pi):
    """! Computes the local homogeneous transformation matrix for the ith robot joint.

    @param i   The robot joint index.
    @param q   The robot angular configuration (i.e. actuators values).
    @param pi  The robot kinematic parameters vector.

    @return  The local homogeneous transformation matrix.
    """

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

def T(q,pi,TBase=TBaseN,TTool=TFlangeN):
    """! Computes the global homogeneous transformation matrix between the world reference frame and the robot flange.

    @param q        The robot angular configuration (i.e. actuators values).
    @param pi       The robot kinematic parameters vector.
    @param TBase    The homogeneous transformation matrix between the world reference frame and the robot base.
    @param TFalnge  The homogeneous transformation matrix between the last robot joint and the robot flange.

    @return  The global homogeneous transformation matrix.
    """

    T = np.eye(4)
    for i in range(JOINTS_NUMBER):
        T = np.dot(T,Ti(i,q,pi)[0])
    return(np.dot(np.dot(TBase,T),TTool))

### Robot identification jacobian matrix

def dTk(k,q,pi,TBase=TBaseN,TTool=np.dot(TFlangeN,TToolN_body)):

    #print("Parameter index = " + str(k))

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
        translationIndex = LINKS_TRANSLATIONS[i][k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i]]
        #print("Link translation :\n" + str(HT[translationIndex - 1]))
        #print("axis : " + str(translationIndex))
        dTk = np.dot(dTk,HT[translationIndex - 1])
    
    else:
        rotationIndex = LINKS_ROTATIONS[i][k - PARAMETERS_INDEX[i] - JOINTS_OFFSET[i] - size(LINKS_TRANSLATIONS[i])]
        #print("Link rotation :\n" + str(HR[rotationIndex - 1]))
        #print("axis : "+str(rotationIndex))
        dTk = np.dot(dTk,HR[rotationIndex - 1])

    for m in sTi[k-PARAMETERS_INDEX[i]+2:]:
        #print("Following transformation matrix :\n" + str(m))
        dTk = np.dot(dTk,m)

    for j in range(i+1,JOINTS_NUMBER):
        #print("Following joint : " + str(j+1))
        dTk = np.dot(dTk,Ti(j,q,pi)[0])

    return(np.dot(np.dot(TBase,dTk),TTool))

def Jpi(q,pi,TBase=TBaseN,TTool=np.dot(TFlangeN,TToolN_body)):
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

def JpiT(q,pi,TBase=TBaseN,TTool=np.dot(TFlangeN,TToolN_body)):
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