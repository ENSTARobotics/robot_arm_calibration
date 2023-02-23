#!/usr/bin/python3

#System packages
import os
import dill

#Utility packages
import sympy as sp
import numpy as np

### Import robot and marker parameters
from ParametersTools import importParameters
importParameters()

### Get robot config directory path
configPathRobot = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2]) + "/config/nominalModels" + "/" + robotName + "/"

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
        return(sp.rot_axis1(angle).T)
    elif(axisIndex == 2):
        return(sp.rot_axis2(angle).T)
    elif(axisIndex == 3):
        return(sp.rot_axis3(angle).T)
    else:
        return(sp.Identity(3))

#If the analytic model does not exist, it is generated
if(not os.path.exists(configPathRobot + "Jacobian_" + robotName + "_" + markerName + ".pickle") or not os.path.exists(configPathRobot + "GeometricModel_" + robotName + "_" + markerName + ".pickle")):

    #import time
    #t0 = time.time()

    ### Global symbols

    q = sp.MatrixSymbol("q",JOINTS_NUMBER,1).as_mutable()

    ### Robot geometric model (revolute joints only)

    #Indices of the first joint/link i parameter in parameters vector
    PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER+1)])

    #Tool function to compute the local transformation and parameters associated to the ith (joint,link)
    def Ti(i):

        T = sp.Identity(4)
        pi = []

        #Joint rotation and offset
        Ti = sp.Identity(4).as_mutable()
        Ti[:3,:3] = rot_axis(np.sign(JOINTS_AXIS[i])*q[i,0],np.abs(JOINTS_AXIS[i]))
        T = T*Ti

        if(JOINTS_OFFSET[i] == 1):
            dqi = sp.symbols("dq"+str(i))
            Ti = sp.Identity(4).as_mutable()
            Ti[:3,:3] = rot_axis(dqi,np.abs(JOINTS_AXIS[i]))
            T = T*Ti
            pi.append(dqi)

        #Link translation
        Taxisi = sp.symbols("Tx"+str(i)+",Ty"+str(i)+",Tz"+str(i))
        for k,axis in enumerate(LINKS_TRANSLATIONS[i]):
            Ti = sp.Identity(4).as_mutable()
            Ti[np.abs(axis)-1,-1] = [[np.sign(axis)*Taxisi[np.abs(axis)-1]]]
            T = T*Ti
            pi.append(Taxisi[np.abs(axis)-1])

        #Link rotation
        Raxisi = sp.symbols("Rx"+str(i)+",Ry"+str(i)+",Rz"+str(i))
        for k,axis in enumerate(LINKS_ROTATIONS[i]):
            Ti = sp.Identity(4).as_mutable()
            Ti[:3,:3] = rot_axis(np.sign(axis)*Raxisi[np.abs(axis)-1],np.abs(axis))
            T = T*Ti
            pi.append(Raxisi[np.abs(axis)-1])

        return(T,pi)

    ### Robot base and tools

    PBase = sp.MatrixSymbol("PBase",3,1).as_mutable()
    PhiBase = sp.MatrixSymbol("PhiBase",3,1).as_mutable()

    TBase = sp.Identity(4).as_mutable()
    TBase[:3,:3] = rot_axis(PhiBase[2,0],3)*rot_axis(PhiBase[1,0],2)*rot_axis(PhiBase[0,0],1)
    TBase[:3,3] = PBase

    TTool = []
    for i,_ in enumerate(TToolN_marker):
        TTooli = sp.ones(4,1).as_mutable()
        TTooli[:3,0] = sp.MatrixSymbol("PTool"+str(i),3,1).as_mutable()
        TTool.append(TTooli)

    TTool = sp.Matrix.hstack(*TTool)
    PTool = TTool[:3,:].T.reshape(3*len(TToolN_marker),1)

    pi = []

    print("Computing geometric model...")
    T = TBase
    for i in range(JOINTS_NUMBER):
        Tii,pii = Ti(i)
        T = T*Tii
        pi.extend(pii)
    T = T*TTool
    P = T[:3,:].T.reshape(3*len(TToolN_marker),1)
    print("Computation done !")

    pi = sp.Matrix(pi)
    parameters = sp.Matrix.vstack(PBase,PhiBase,pi,TTool[:3,:].T.reshape(3*len(TToolN_marker),1))

    print("Computing jacobian...")
    J = P.jacobian(parameters)
    print("Computation done !")

    dill.settings['recurse'] = True

    print("Turning sympy expressions to lambda functions...")

    Pfunc = sp.lambdify((parameters,q),P,modules=["numpy","scipy"])
    with open(configPathRobot + "GeometricModel_" + robotName + "_" + markerName + ".pickle", "wb+") as file:
        dill.dump(Pfunc, file)
    Jfunc = sp.lambdify((parameters,q),J,modules=["numpy","scipy"])
    with open(configPathRobot + "Jacobian_" + robotName + "_" + markerName + ".pickle", "wb+") as file:
        dill.dump(Jfunc, file)

    print("Done !")   

    #print("-- Analytic model computed in " + str(time.time() - t0) + " seconds --")

#If the analytic model exists, it is loaded
print("Loading geometric model and jacobian...")
rawJfunc = dill.load(open(configPathRobot + "Jacobian_" + robotName + "_" + markerName + ".pickle", "rb"))
rawPfunc = dill.load(open(configPathRobot + "GeometricModel_" + robotName + "_" + markerName + ".pickle", "rb"))
print("Done !")

#Formatting analytic functions with proper output shapes
PARAMETERS_NUMBER = 6 + 3*len(TToolN_marker) + len(pi0)

def Jfunc(parameters,q):
    J = rawJfunc(parameters,q)
    for i,row in enumerate(J):
        for j,item in enumerate(row):
            try:
                J[i,j] = J[i,j].item()
            except:
                J[i,j] = float(J[i,j])
    return(J)

def Pfunc(parameters,q):
    P = rawPfunc(parameters,q)
    for i,item in enumerate(P):
        try:
            P[i] = P[i].item()
        except:
            P[i] = float(P[i])
    return(P)