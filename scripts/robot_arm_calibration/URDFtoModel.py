#!/usr/bin/python3

#System packages
import os
import sys
import yaml

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
from urchin import URDF 

#Custom package
from robot_arm_calibration.URDFTools import getURDF, getSortedJoints, size

def URDFtoModel(URDFFilePath, flangeLinkName):
    """! Converts an URDF description to identifiable geometric parameters.

    @param URDFFilePath     The path to the URDF file
    @param flangeLinkName   The name of the robot's flange link.

    @return  The identifiable geometric parameters.
    """

    #Load URDF file
    robot = getURDF(URDFFilePath)

    #Get URDF joints in the right order
    baseJoints,actuatedJoints,flangeJoints = getSortedJoints(robot,flangeLinkName)

    ### Create model

    #Number of revolute joints
    JOINTS_NUMBER = len(actuatedJoints)

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

    #Compute robot base joint transformation
    TBaseInter = np.eye(4)
    for joint in baseJoints:
        globalT = np.dot(globalT,joint.origin)
        TBaseInter = np.dot(TBaseInter,joint.origin)

    ### Get robot identifiable parameters distribution (joint offsets, link translations & rotations)
    axisNorm = all(all(joint.axis == np.array([0,0,1])) for joint in actuatedJoints)
    for i,joint in enumerate(actuatedJoints):

        deltaP = np.dot(globalT,joint.origin)[:3,-1] - globalT[:3,-1]
        globalT = np.dot(globalT,joint.origin)

        #Retrive current joint axis - ej (completed by uj and vj to form an orthonormal frame)
        if(axisNorm):
            currentjointIndex = np.where(np.round(np.abs(globalT[:3,2]),4)==1)[0][0]
            JOINTS_AXIS.append(int(np.sign(globalT[:3,2][currentjointIndex])*(currentjointIndex + 1)))
        else:
            currentjointIndex = np.where(np.round(np.abs(joint.axis),4)==1)[0][0]
            JOINTS_AXIS.append(int(np.sign(joint.axis[currentjointIndex])*(currentjointIndex + 1)))

        #Get identifiable links translations and rotations - Tej/uj/vj and Rej/uj/vj
        
        #Remove redundant parameters from previous links
        if(i > 0):
            #If ej-1 // ej, we delete Tuj-k/vj-k which defines the translation orthogonal to ej, with k minimal
            if(abs(JOINTS_AXIS[-1]) == abs(JOINTS_AXIS[-2])):
                k = 1
                while(k < len(JOINTS_AXIS)):
                    if((abs(JOINTS_AXIS[-1])+1)%3 in LINKS_TRANSLATIONS[-k] or (abs(JOINTS_AXIS[-1])+2)%3 in LINKS_TRANSLATIONS[-k]):
                        break
                    else:
                        k+=1
                LINKS_TRANSLATIONS[-k] = np.delete(LINKS_TRANSLATIONS[-k],np.argmin(deltaP[np.where(LINKS_TRANSLATIONS[-k] != abs(JOINTS_AXIS[-1]))]))

            #If ej-1 _|_ ej, we delete Ruj-1/vj-1 which corresponds to Rej
            else:
                LINKS_ROTATIONS[-1] = np.delete(LINKS_ROTATIONS[-1],np.where(LINKS_ROTATIONS[-1] == abs(JOINTS_AXIS[-1]))[0][0])

        #Exclude last link: it will be handled with flange joint/link
        if(i == JOINTS_NUMBER - 1):

            #Delete redundant link translation colinear to the last joint axis, and add it to the flange
            if(abs(JOINTS_AXIS[-1]) in LINKS_TRANSLATIONS[-1]):
                LINKS_TRANSLATIONS[-1] = np.delete(LINKS_TRANSLATIONS[-1],np.where(LINKS_TRANSLATIONS[-1] == abs(JOINTS_AXIS[-1]))[0][-1])
            
            LINKS_TRANSLATIONS.append(np.array([]))
            LINKS_ROTATIONS.append(np.array([]))

        #Handle all the other joints
        else:
            #Add translations and rotations along all axis except the current joint axis
            LINKS_TRANSLATIONS.append(np.sort(np.array([(currentjointIndex + 1)%3 + 1,(currentjointIndex + 2)%3 + 1])))
            LINKS_ROTATIONS.append(np.sort(np.array([(currentjointIndex + 1)%3 + 1,(currentjointIndex + 2)%3 + 1])))

    #Compute robot flange joint transformation
    TFlangeInter = np.eye(4)
    for joint in flangeJoints:
        TFlangeInter = np.dot(TFlangeInter,joint.origin) 

    LINKS_TRANSLATIONS = [item.tolist() for item in LINKS_TRANSLATIONS]
    LINKS_ROTATIONS = [item.tolist() for item in LINKS_ROTATIONS]

    ### Get robot nominal parameters
    pi0 = []

    #Trick to avoid left behind translations (c.f. following loop)
    needFill = []

    #Set nominal robot base transformation
    TBaseN = actuatedJoints[0].origin   #Default : robot base joint is aligned on the reference world frame

    #Set robot nominal values for identifiable parameters
    globalT = np.dot(TBaseInter,actuatedJoints[0].origin)

    deltaFlange = np.zeros(3)
    for i,joint in enumerate(actuatedJoints[1:]):

        #Joints rotation offsets
        if(JOINTS_OFFSET[i] != 0):
            pi0.append(0)

        deltaP = np.dot(globalT,joint.origin)[:3,-1] - globalT[:3,-1]
        globalT = np.dot(globalT,joint.origin)

        #If we have not reached the flange, we can still redistribute translations (all flange translations and rotations will be merged with the tool transformation)
        if(i != len(actuatedJoints)-2):

            #If all identifiable link translation parameters do not have a corresponding non-zero nominal value, they might be found later (except if the next link is the last - excluded - one) ! For now, we pin it somewhere.
            fillable = np.where(np.abs(np.array([deltaP[j-1] for j in LINKS_TRANSLATIONS[i]])) < 1e-5)[0]
            if(len(fillable) != 0):
                needFill.append([i,[LINKS_TRANSLATIONS[i][k] for k in fillable]])

            #A non-zero nominal value may have been left behind (i.e. not associated to an identifiable link translation parameter). It may be related to a previously pined link translation parameter.
            for j in needFill:
                for n,k in enumerate(j[1]):
                    if(k not in LINKS_TRANSLATIONS[i] and np.abs(deltaP[k-1]) > 1e-5):
                        pi0[sum(JOINTS_OFFSET[:j[0]+1]) + size(LINKS_TRANSLATIONS[:j[0]]) + size(LINKS_ROTATIONS[:j[0]]) + n] = np.round(deltaP[k-1],6)
                        needFill.remove(j)

        #Links translations and rotations
        pi0.extend([np.round(deltaP[i-1],10) for i in LINKS_TRANSLATIONS[i]])
        pi0.extend([0.0 for i in LINKS_ROTATIONS[i]])

    #Set nominal robot flange transformation
    deltaP += np.dot(globalT,TFlangeInter)[:3,-1] - globalT[:3,-1] + deltaFlange
    globalT = np.dot(globalT,TFlangeInter)
    TFlangeN = np.eye(4)
    TFlangeN[:3,:3] = globalT[:3,:3]
    TFlangeN[:3,-1][np.abs(JOINTS_AXIS[-1]) - 1] = deltaP[np.abs(JOINTS_AXIS[-1]) - 1]

    #DEPRECATED
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

    #DEBUG
    #print(LINKS_ROTATIONS)
    #print(LINKS_TRANSLATIONS)
    #print(pi0)
    #print(TBaseN)
    #print(TFlangeN)

    return(JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN)
