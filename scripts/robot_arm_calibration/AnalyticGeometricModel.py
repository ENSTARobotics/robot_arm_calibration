#!/usr/bin/python3

#System packages
import os
import sys
import time

#Utility packages
import casadi as cs
from liecasadi import SO3
import numpy as np

#Robot parameters tools
from robot_arm_calibration.ParametersTools import getPackagePath, importParameters, importAllParametersFromFile, getIdentifiableParametersPartial, getIdentifiableParametersFull

### Tool functions
def size(L):
    """! Computes the size of a nested list as the sum of the sizes of all contained lists

    @param L   The nested list.

    @return  The size of the nested list.
    """
    if type(L) == list or type(L) == np.ndarray:
        return(sum(size(l) for l in L))
    else:
        return(1) 

def rot_axis(angle,axisIndex):
    """! Computes the analytic matrix representation of a rotation given an angle and an axis.

    @param angle        The analytic rotation angle.
    @param axisIndex    The index of the rotation axis : 1 -> X, 2 -> Y, 3 -> Z.

    @return  The analytic matrix representation of the rotation.
    """
    if(axisIndex == 1):
        return(SO3.qx(angle).as_matrix())
    elif(axisIndex == 2):
        return(SO3.qy(angle).as_matrix())
    elif(axisIndex == 3):
        return(SO3.qz(angle).as_matrix())
    else:
        return(cs.SX.eye(4))

def quat_to_mrp(quat):
    """! Converts a casadi quaternion representation of a rotation into a scipy modified Rodrigues parameters representation

    @param quat   The casadi quaternion representation of a rotation.
    @return       The scipy modified Rodrigues parameters representation of the rotation.
    """

    quat_matrix = quat.xyzw
    #sign = -1 if quat_matrix[-1] < 0 else 1
    sign = quat_matrix[-1]/cs.fabs(quat_matrix[-1])
    denominator = 1 + sign * quat_matrix[-1]
    return((sign/denominator) * cs.vertcat(quat_matrix[0],quat_matrix[1],quat_matrix[2]))

def mrp_to_matrix(mrp):
    """! Converts a scipy modified Rodrigues parameters representation of a rotation into a casadi rotation matrix

    @param mrp   The scipy modified Rodrigues parameters representation of a rotation.
    @return      The casadi quaternion representation of the rotation.
    """

    mrp_squared_plus_1 = cs.dot(mrp,mrp) + 1
    x = 2*mrp[0]/mrp_squared_plus_1
    y = 2*mrp[1]/mrp_squared_plus_1
    z = 2*mrp[2]/mrp_squared_plus_1
    w = (2 - mrp_squared_plus_1)/mrp_squared_plus_1
    return(SO3.from_quat(cs.vertcat(x,y,z,w)).as_matrix())

#TODO 
#1) Non cartesian aligned axes (ie, not X(1),Y(2),Z(3) aligned)
#2) Joints flexibilities

### Default => Get model from parameters file
try:
    importAllParametersFromFile(sys.argv[1])

    ### Get robot config directory path
    configPathRobot = os.path.dirname(sys.argv[1]) + "/"

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

    ### Get robot config directory path
    configPathRobot = getPackagePath() + "/config/nominalModels/robots/" + robotName + "/"

fileID = str(robotName) + "_" + str(toolName) if (len(toolName) != 0) else str(robotName)

### Global symbols

q = cs.SX.sym('q',JOINTS_NUMBER)

### Robot geometric model (revolute joints only)

#Indices of the first joint/link i parameter in parameters vector
PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER+1)])

#Tool function to compute the local transformation and parameters associated to the ith (joint,link)
def Ti(i):

    T = cs.SX.eye(4)
    pi = []

    #Joint rotation and offset
    Ti = cs.SX.eye(4)
    Ti[:3,:3] = rot_axis(np.sign(JOINTS_AXIS[i])*q[i],np.abs(JOINTS_AXIS[i]))
    T = T@Ti

    if(JOINTS_OFFSET[i] == 1):
        dqi = cs.SX.sym("dq"+str(i))
        Ti = cs.SX.eye(4)
        Ti[:3,:3] = rot_axis(dqi,np.abs(JOINTS_AXIS[i]))
        T = T@Ti
        pi.append(dqi)

    #Link translation
    Taxisi = [cs.SX.sym("Tx"+str(i)),cs.SX.sym("Ty"+str(i)),cs.SX.sym("Tz"+str(i))]
    for k,axis in enumerate(LINKS_TRANSLATIONS[i]):
        Ti = cs.SX.eye(4)
        Ti[np.abs(axis)-1,-1] = np.sign(axis)*Taxisi[np.abs(axis)-1]
        T = T@Ti
        pi.append(Taxisi[np.abs(axis)-1])

    #Link rotation
    Raxisi = [cs.SX.sym("Rx"+str(i)),cs.SX.sym("Ry"+str(i)),cs.SX.sym("Rz"+str(i))]
    for k,axis in enumerate(LINKS_ROTATIONS[i]):
        Ti = cs.SX.eye(4)
        Ti[:3,:3] = rot_axis(np.sign(axis)*Raxisi[np.abs(axis)-1],np.abs(axis))
        T = T@Ti
        pi.append(Raxisi[np.abs(axis)-1])

    return(T,pi)

### Robot base and tools

PBase = cs.SX.sym('PBase',3)
PhiBase = cs.SX.sym('PhiBase',3)

TBase = cs.SX.eye(4)
TBase[:3,:3] = mrp_to_matrix(PhiBase)
TBase[:3,3] = PBase

#Multiple tool transformation matrices => Partial pose measurement
if(len(TToolN) > 1):
    TTool = []
    for i,_ in enumerate(TToolN):
        TTooli = cs.SX.ones(4,1)
        TTooli[:3,0] = cs.SX.sym("PTool"+str(i),3,1)
        TTool.append(TTooli)
    TTool = cs.horzcat(*TTool)

#Single tool transformation matrix => Full pose measurement
else:
    PTool = cs.SX.sym('PTool',3)
    PhiTool = cs.SX.sym('PhiTool',3)

    TTool = cs.SX.eye(4)
    TTool[:3,:3] = mrp_to_matrix(PhiTool)
    TTool[:3,3] = PTool

pi = []

T = TBase
Traw = cs.SX.eye(4)
for i in range(JOINTS_NUMBER):
    Tii,pii = Ti(i)
    T = T@Tii
    Traw = Traw@Tii
    pi.extend(pii)
T = T@TTool

if(len(TToolN) > 1):
    parameters = cs.vertcat(PBase,PhiBase,*pi,TTool[:3,:].reshape((3*len(TToolN),1)))
else:
    parameters = cs.vertcat(PBase,PhiBase,*pi,PTool,PhiTool)

if(len(TToolN) > 1):
    P = T[:3,:].reshape((3*len(TToolN),1))
else:
    P = cs.vertcat(T[:3,3],quat_to_mrp(SO3.from_matrix(T[:3,:3])))

# tic = time.perf_counter()
J = cs.jacobian(P,parameters)
# tac = time.perf_counter()
# print("Elapsed time computation: ", tac-tic)

# tic = time.perf_counter()
Tfunc = cs.Function('T',[parameters,q],[Traw])
Pfunc = cs.Function('P',[parameters,q],[P])
Jfunc = cs.Function('J',[parameters,q],[J])
# tac = time.perf_counter()
# print("Elapsed time conversion: ", tac-tic) 

PARAMETERS_NUMBER = 6 + 3*len(TToolN) + len(piN) if(len(TToolN) > 1) else 6 + 6 + len(piN)

# q = np.ones(JOINTS_NUMBER)*np.pi/2
# np.set_printoptions(edgeitems=30, linewidth=100000, 
#    formatter=dict(float=lambda x: "%.3g" % x))
# print(np.array(Tfunc(getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN),q)))
# print(np.array(Pfunc(getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN),q)))
# print(np.array(Jfunc(getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN),q)))

# T = []
# for i in range(10):
#     q = np.random.random(JOINTS_NUMBER)
#     print(q)
#     tic = time.perf_counter()
#     print(Jfunc(getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN),q))
#     tac = time.perf_counter()
#     print("Elapsed time evaluation: ", tac-tic)
#     T.append(tac-tic)
# print("Average elapsed time evaluation: ", sum(T)/len(T))

# from scipy.spatial.transform import Rotation as R
# rotation = R.from_matrix(np.eye(3))
# print(rotation.as_mrp())
# print(rotation.as_quat())

# rotation_casadi = mrp_to_matrix(rotation.as_mrp())
# q_casadi = SO3.from_matrix(rotation_casadi).as_quat()
# print(q_casadi)
# mrp_casadi = quat_to_mrp(q_casadi)
# print(mrp_casadi)

# print(rot_axis(np.pi/2,1))