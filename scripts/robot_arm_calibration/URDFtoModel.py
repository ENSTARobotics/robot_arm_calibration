#!/usr/bin/python3

#System packages
import os
import sys
import yaml
import subprocess
import re
import glob

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

def size(L):
    """! Computes the size of a nested list as the sum of the sizes of all contained lists

    @param L   The nested list.

    @return  The size of the nested list.
    """
    if type(L) == list or type(L) == np.ndarray:
        return(sum(size(l) for l in L))
    else:
        return(1) 

def URDFtoModel(urdfFilePath):
    """! Converts an URDF description to identifiable geometric parameters.

    @param urdfFilePath The path to the URDF file

    @return  The identifiable geometric parameters.
    """

    ### Load URDF file

    #XACRO file
    if(urdfFilePath.split(".")[-1] == "xacro"):
        xacro = "echo $(xacro " + urdfFilePath + ")"
        xacroProcess = subprocess.run(xacro,shell=True,stdout=subprocess.PIPE)
        xacroOutput = str(xacroProcess.stdout)[2:-3]

        packages = np.unique(np.array([xacroOutput[int(m.start()):int(m.end())] for m in re.finditer('package://[a-zA-Z_]+/', xacroOutput)]))

        for package in packages:
            rosFind = "rospack find " + package.split("/")[-2]
            rosFindProcess = subprocess.run(rosFind,shell=True,stdout=subprocess.PIPE)
            rosFindOutput = str(rosFindProcess.stdout)[2:-3] + "/" 
            xacroOutput = xacroOutput.replace(package,rosFindOutput)

        xacroOutput = xacroOutput.replace("file://","")

        URDFfile = open("/tmp/tmp.urdf", "w")
        URDFfile.write(xacroOutput)
        URDFfile.close()

        ### Modify URDF file
        robot = URDF.load('/tmp/tmp.urdf')

    #URDF file
    else:
        robot = URDF.load(urdfFilePath)

    #Number of revolute joints
    JOINTS_NUMBER = len(robot.actuated_joints)

    #Angular offset at each joint (yes (1) or no (0))
    JOINTS_OFFSET = np.ones(JOINTS_NUMBER)
    JOINTS_OFFSET[0] = 0
    JOINTS_OFFSET[-1] = 0
    JOINTS_OFFSET = JOINTS_OFFSET.astype(int).tolist()

    #Joints axis index (x (1), y (2) or z (3)) 
    JOINTS_AXIS = []

    #Links translations axis (along x (1), y (2) or z (3))  
    LINKS_TRANSLATIONS = []

    #Links rotations axis (around x (1), y (2) or z (3))   
    LINKS_ROTATIONS = []

    globalT = np.eye(4)

    #Retreive intermediate conventionnal joints if needed
    TBaseInter = np.eye(4)
    TFlangeInter = np.eye(4)
    if(len(robot.joints) != len(robot.actuated_joints)):
        for joint in robot.joints:
            #Investigating not actuated intermediate joints...
            if(joint not in robot.actuated_joints):
                #Conventionnal base joints (e.g. "base_inertial")
                TBaseInter = np.dot(TBaseInter,joint.origin)
            else:
                break
        #Needed kinematic chain closing joint !
        if(robot.joints[-1] not in robot.actuated_joints):
            TFlangeInter = np.dot(TFlangeInter,robot.joints[-1].origin)
    globalT = TBaseInter

    ### Robot joints
    for i,joint in enumerate(robot.actuated_joints):

        deltaP = np.dot(globalT,joint.origin)[:3,-1] - globalT[:3,-1]
        globalT = np.dot(globalT,joint.origin)

        #Retrive current joint axis - ej (completed by uj and vj to form an orthonormal frame)
        currentjointIndex = np.where(np.abs(globalT[:3,2])==1)[0][0]
        JOINTS_AXIS.append(int(np.sign(globalT[:3,2][currentjointIndex])*(currentjointIndex + 1)))

        #Building links translations and rotations - Tej/uj/vj and Rej/uj/vj

        #Base joint
        if(i == 0):
            TBaseN = np.eye(4)
            TBaseN[:3,:3] = globalT[:3,:3]
            TBaseN[:3,-1][currentjointIndex] = deltaP[currentjointIndex]
        
        #Other joints
        else:
            #If ej-1 // ej, we delete Tuj-k/vj-k which defines the translation orthogonal to ej, with k minimal
            if(abs(JOINTS_AXIS[-1]) == abs(JOINTS_AXIS[-2])):
                k = 1
                while(abs(JOINTS_AXIS[-1]) in LINKS_TRANSLATIONS[-k]):
                    k+=1
                LINKS_TRANSLATIONS[-k] = np.delete(LINKS_TRANSLATIONS[-k],np.where(LINKS_TRANSLATIONS[-k] != abs(JOINTS_AXIS[-1]))[0][-1])

            #If ej-1 _|_ ej, we delete Ruj-1/vj-1 which corresponds to Rej
            else:
                LINKS_ROTATIONS[-1] = np.delete(LINKS_ROTATIONS[-1],np.where(LINKS_ROTATIONS[-1] == abs(JOINTS_AXIS[-1]))[0][0])

        #Flange joint
        if(i == JOINTS_NUMBER - 1):
            deltaP += np.dot(globalT,TFlangeInter)[:3,-1] - globalT[:3,-1]
            globalT = np.dot(globalT,TFlangeInter)
            
            TFlangeN = np.eye(4)
            TFlangeN[:3,:3] = globalT[:3,:3]
            TFlangeN[:3,-1][currentjointIndex] = deltaP[currentjointIndex]

            if(abs(JOINTS_AXIS[-1]) in LINKS_TRANSLATIONS[-1]):
                LINKS_TRANSLATIONS[-1] = np.delete(LINKS_TRANSLATIONS[-1],np.where(LINKS_TRANSLATIONS[-1] == abs(JOINTS_AXIS[-1]))[0][-1])
            
            LINKS_TRANSLATIONS.append(np.array([]))
            LINKS_ROTATIONS.append(np.array([]))

        #Other joints
        else:
            LINKS_TRANSLATIONS.append(np.sort(np.array([(currentjointIndex + 1)%3 + 1,(currentjointIndex + 2)%3 + 1])))
            LINKS_ROTATIONS.append(np.sort(np.array([(currentjointIndex + 1)%3 + 1,(currentjointIndex + 2)%3 + 1])))

    LINKS_TRANSLATIONS = [item.tolist() for item in LINKS_TRANSLATIONS]
    LINKS_ROTATIONS = [item.tolist() for item in LINKS_ROTATIONS]

    #Robot nominal parameters
    pi0 = []

    globalT = np.dot(TBaseInter,robot.actuated_joints[0].origin)

    needFill = []

    for i,joint in enumerate(robot.actuated_joints[1:]):

        if(JOINTS_OFFSET[i] != 0):
            pi0.append(0)

        deltaP = np.dot(globalT,joint.origin)[:3,-1] - globalT[:3,-1]
        globalT = np.dot(globalT,joint.origin)

        if(i != len(robot.actuated_joints)-2 and (np.array([deltaP[i-1] for i in LINKS_TRANSLATIONS[i]]) == 0).all()):
            needFill.append(i)
        else:
            for j in needFill:
                for k in LINKS_TRANSLATIONS[j]:
                    if(k not in LINKS_TRANSLATIONS[i] and np.abs(deltaP[k-1]) > 1e-10):
                        pi0[sum(JOINTS_OFFSET[:j]) + size(LINKS_TRANSLATIONS[:j]) + size(LINKS_ROTATIONS[:j]) + k - 1] = np.round(deltaP[k-1],10)
                        del needFill[j]

        pi0.extend([np.round(deltaP[i-1],10) for i in LINKS_TRANSLATIONS[i]])
        pi0.extend([0.0 for i in LINKS_ROTATIONS[i]])

    #print("Saving geometric model parameters...")
    #with open(os.path.dirname(os.path.realpath(__file__)) + "/" + robotName + "/" + robotName + #"_parameters.npy","wb+") as file:
    #    np.save(file,JOINTS_NUMBER)
    #    np.save(file,JOINTS_AXIS)
    #    np.save(file,JOINTS_OFFSET)
    #    np.save(file,LINKS_TRANSLATIONS)
    #    np.save(file,LINKS_ROTATIONS)
    #    np.save(file,pi0)
    #    np.save(file,TBaseN)
    #    np.save(file,TFlangeN)
    #print("Done !")

    #print("-- Model parameters computed in " + str(time.time() - t0) + " seconds --")

    return(JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN)