#!/usr/bin/python3

#System packages
import os
import sys

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

#Time package
import time

def DHtoModel(DHParameters, robotName):
    """! Converts Denavit-Hartenberg parameters to identifiable geometric parameters.

    @param DHParameters   The modified Denavit-Hartenberg parameters (alpha, d, r, theta)
    @param robotName      The name of the robot

    @return  The identifiable geometric parameters.
    """

    #Starting timer
    t0 = time.time()

    ### Building model parameters
    print("Building robot geometric model parameters !")

    #Number of revolute joints
    JOINTS_NUMBER = len(DHParameters)

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

    localT = lambda alpha,d,r,theta : np.array([[np.cos(theta),-np.sin(theta),0,d],
                                                [np.cos(alpha)*np.sin(theta),np.cos(alpha)*np.cos(theta),-np.sin(alpha),-r*np.sin(alpha)],
                                                [np.sin(alpha)*np.sin(theta),np.sin(alpha)*np.cos(theta),np.cos(alpha),r*np.cos(alpha)],
                                                [0,0,0,1]])

    globalT = np.eye(4)

    for j,parameters in enumerate(DHParameters):
        deltaP = np.dot(globalT,localT(*parameters))[:3,-1] - globalT[:3,-1]
        globalT = np.dot(globalT,localT(*parameters))

        #Retrive current joint axis - ej (completed by uj and vj to form an orthonormal frame)
        currentjointIndex = np.where(np.abs(globalT[:3,2])==1)[0][0]
        JOINTS_AXIS.append(int(np.sign(globalT[:3,2][currentjointIndex])*(currentjointIndex + 1)))

        #Building links translations and rotations - Tej/uj/vj and Rej/uj/vj

        #Base link
        if(j == 0):
            TBaseN = np.eye(4)
            TBaseN[:3,-1][currentjointIndex] = deltaP[currentjointIndex]
        
        #Other links
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

        #Flange link
        if(j == JOINTS_NUMBER - 1):
            TFlangeN = np.eye(4)
            TFlangeN[:3,:3] = globalT[:3,:3]
            TFlangeN[:3,-1][currentjointIndex] = deltaP[currentjointIndex]

            if(abs(JOINTS_AXIS[-1]) in LINKS_TRANSLATIONS[-1]):
                LINKS_TRANSLATIONS[-1] = np.delete(LINKS_TRANSLATIONS[-1],np.where(LINKS_TRANSLATIONS[-1] == abs(JOINTS_AXIS[-1]))[0][-1])
            
            LINKS_TRANSLATIONS.append(np.array([]))
            LINKS_ROTATIONS.append(np.array([]))

        #Other links
        else:
            LINKS_TRANSLATIONS.append(np.sort(np.array([(currentjointIndex + 1)%3 + 1,(currentjointIndex + 2)%3 + 1])))
            LINKS_ROTATIONS.append(np.sort(np.array([(currentjointIndex + 1)%3 + 1,(currentjointIndex + 2)%3 + 1])))

    LINKS_TRANSLATIONS = [item.tolist() for item in LINKS_TRANSLATIONS]
    LINKS_ROTATIONS = [item.tolist() for item in LINKS_ROTATIONS]

    #Robot nominal parameters
    pi0 = []

    globalT = localT(*DHParameters[0])

    for j,parameters in enumerate(DHParameters[1:]):

        if(JOINTS_OFFSET[j] != 0):
            pi0.append(0)

        deltaP = np.dot(globalT,localT(*parameters))[:3,-1] - globalT[:3,-1]
        globalT = np.dot(globalT,localT(*parameters))

        pi0.extend([deltaP[i-1] for i in LINKS_TRANSLATIONS[j]])
        pi0.extend([0.0 for i in LINKS_ROTATIONS[j]])

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

    print("-- Model parameters computed in " + str(time.time() - t0) + " seconds --")

    return(JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,pi0,TBaseN,TFlangeN)
