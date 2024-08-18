#!/usr/bin/python3

#System packages
import sys
import time

#Utility packages
#from jax import jit    #Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once. Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.
import jax.numpy as np
from jax.scipy.spatial.transform import Rotation as R

#Jacobian computation package
from jax import jacfwd, jacrev  #jacfwd uses forward-mode automatic differentiation, which is more efficient for “tall” Jacobian matrices, while jacrev uses reverse-mode, which is more efficient for “wide” Jacobian matrices. For matrices that are near-square, jacfwd probably has an edge over jacrev.

from robot_arm_calibration.ParametersTools import getIdentifiableParametersPartial, getIdentifiableParametersFull

def size(L):
    """! Computes the size of a nested list as the sum of the sizes of all contained lists

    @param L   The nested list.

    @return  The size of the nested list.
    """
    if type(L) == list or type(L) == np.ndarray:
        return(sum(size(l) for l in L))
    else:
        return(1) 

from robot_arm_calibration.ParametersTools import importParameters, importAllParametersFromFile

### Default => Get model from parameters file
try:
    importAllParametersFromFile(sys.argv[1])

except:
    ###Get model from robot and tool name
    robotName = ""
    toolName = ""

    try:
        robotName = sys.argv[1]
    except:
        pass
    try:
        toolName = sys.argv[2]
    except:
        pass

    importParameters(robotName,toolName)

### Robot geometric model (revolute joints only)

#Indices of the first joint/link i parameter in parameters vector
PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER+1)])

def Ti(i,q,pi):
    """! Computes the local homogeneous transformation matrix for the ith robot joint.

    @param i   The robot joint index.
    @param q   The robot angular configuration (i.e. actuators values).
    @param pi  The robot kinematic parameters vector.

    @return  The local homogeneous transformation matrix, the list of the involved successive primitive transformations (single axis translation/rotation) matrices.
    """

    T = np.eye(4)

    parameters = pi[PARAMETERS_INDEX[i]:PARAMETERS_INDEX[i+1]]

    ###Joint rotation and offset
    Ti = np.eye(4)
    angles = np.zeros(3)

    angles = angles.at[np.abs(JOINTS_AXIS[i])-1].set(np.sign(JOINTS_AXIS[i])*q[i])
    Ti = Ti.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())
    T = np.dot(T,Ti)

    if(JOINTS_OFFSET[i] == 1):
        Ti = np.eye(4)
        angles = np.zeros(3)

        angles = angles.at[np.abs(JOINTS_AXIS[i])-1].set(parameters[0])
        Ti = Ti.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())
        T = np.dot(T,Ti)

    ###Link translation
    for n,axis in enumerate(LINKS_TRANSLATIONS[i]):
        Ti = np.eye(4)
        Ti = Ti.at[np.abs(axis)-1,-1].set(np.sign(axis)*parameters[JOINTS_OFFSET[i] + n])

        T = np.dot(T,Ti)

    ###Link rotation
    for n,axis in enumerate(LINKS_ROTATIONS[i]):
        Ti = np.eye(4)
        angles = np.zeros(3)

        angles = angles.at[np.abs(axis)-1].set(np.sign(axis)*parameters[JOINTS_OFFSET[i] + len(LINKS_TRANSLATIONS[i]) + n])
        Ti = Ti.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())

        T = np.dot(T,Ti)

    return(T)

def T(q,pi,TBase=TBaseN,TTool=TFlangeN):
    """! Computes the global homogeneous transformation matrix between the world reference frame and the robot tool.

    @param q        The robot angular configuration (i.e. actuators values).
    @param pi       The robot kinematic parameters vector.
    @param TBase    The homogeneous transformation matrix between the world reference frame and the robot base.
    @param TTool    The homogeneous transformation matrix between the last robot joint and the robot tool.

    @return  The global homogeneous transformation matrix.
    """

    T = TBase
    for i in range(JOINTS_NUMBER):
        T = np.dot(T,Ti(i,q,pi))
    return(np.dot(T,TTool))

def P(q,pi,TBase=TBaseN,TTool=TToolN):
    """! Computes the position of end effector points in the world reference frame.

    @param q        The robot angular configuration (i.e. actuators values).
    @param pi       The robot kinematic parameters vector.
    @param TBase    The homogeneous transformation matrix between the world reference frame and the robot base frame.
    @param TTool    The homogeneous transformation matrix between the last robot joint and the robot tool.

    @return  The positions of end effector points
    """

    T = TBase
    for i in range(JOINTS_NUMBER):
        T = np.dot(T,Ti(i,q,pi))
    P = np.vstack([np.dot(T,t)[:3,-1] for t in TTool])
    return(P)

def Tfunc(parameters, q):
    """! Computes the global homogeneous transformation matrix between the robot base frame and the robot flange frame.

    @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
    @param q            The robot angular configuration (i.e. actuators values).

    @return  The global homogeneous transformation matrix between the robot base frame and the robot flange frame.
    """
    
    pi = parameters[6:6+len(piN)]
    T = np.eye(4)
    for i in range(JOINTS_NUMBER):
        T = np.dot(T,Ti(i,q,pi))
    return(T)

def PfuncPartial(parameters, q):
    """! Computes the position of end effector points in the robot base frame.

    @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
    @param q            The robot angular configuration (i.e. actuators values).

    @return  The positions of end effector points in the robot base frame.
    """

    #Robot base
    TBase = np.eye(4)
    TBase = TBase.at[:3,-1].set(parameters[:3])
    TBase = TBase.at[:3,:3].set(R.from_mrp(parameters[3:6]).as_matrix())

    #Robot geometric parameters
    pi = parameters[6:6+len(piN)]

    #Robot flange and tool
    TTool = np.empty(np.shape(TToolN))
    for i,P in enumerate(np.reshape(parameters[6+len(piN):],(-1,3))):
        tmp = np.dot(TFlangeN,TToolN[i])
        tmp = tmp.at[:3,-1].set(P)
        TTool = TTool.at[i].set(tmp)

    T = TBase
    for i in range(JOINTS_NUMBER):
        T = np.dot(T,Ti(i,q,pi))
    P = np.hstack([np.dot(T,t)[:3,-1] for t in TTool])
    return(P)

def PfuncFull(parameters, q):
    """! Computes the global homogeneous transformation matrix between the robot base frame and the robot flange frame.

    @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
    @param q            The robot angular configuration (i.e. actuators values).

    @return  The global homogeneous transformation matrix between the robot base frame and the robot flange frame.
    """
    
    #Robot base
    TBase = np.eye(4)
    TBase = TBase.at[:3,-1].set(parameters[:3])
    TBase = TBase.at[:3,:3].set(R.from_mrp(parameters[3:6]).as_matrix())

    #Robot geometric parameters
    pi = parameters[6:6+len(piN)]

    #Robot flange and tool
    TTool = np.eye(4)
    TTool = TTool.at[:3,-1].set(parameters[6+len(piN):9+len(piN)])
    TTool = TTool.at[:3,:3].set(R.from_mrp(parameters[9+len(piN):]).as_matrix())

    T = TBase
    for i in range(JOINTS_NUMBER):
        T = np.dot(T,Ti(i,q,pi))
    T = np.dot(T,TTool)

    P = np.hstack((T[:3,-1],R.from_matrix(T[:3,:3]).as_mrp()))
    return(P)

if(len(TToolN) > 1):
    Pfunc = PfuncPartial
else:
    Pfunc = PfuncFull

### Robot identification jacobian matrix
# tic = time.perf_counter()
Jfunc = jacfwd(Pfunc)
# tac = time.perf_counter()
# print("Elapsed time computation: ", tac-tic)

PARAMETERS_NUMBER = 6 + 3*len(TToolN) + len(piN) if(len(TToolN) > 1) else 6 + 6 + len(piN)

#np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
#q = np.ones(JOINTS_NUMBER)*np.pi/2

# print(np.array(Tfunc(getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN),q)))
# print(np.array(Pfunc(getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN),q)))
# print(np.array(Jfunc(getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN),q)))

# T = []
# for i in range(10):
#     import numpy.random as random
#     q = random.random(JOINTS_NUMBER)
#     #print(q)
#     tic = time.perf_counter()
#     if(len(TToolN) > 1):
#         Jtest = Jfunc(getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN),q)
#     else:
#         Jtest = Jfunc(getIdentifiableParametersFull(TBaseN,TFlangeN,TToolN,piN),q)
#     #print(Jtest)
#     tac = time.perf_counter()
#     #print("Elapsed time evaluation: ", tac-tic)
#     T.append(tac-tic)
# print("Average elapsed time evaluation: ", sum(T)/len(T))

