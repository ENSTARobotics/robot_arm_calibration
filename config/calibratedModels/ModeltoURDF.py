#!/usr/bin/python3

#System packages
import os
import sys
import yaml
import subprocess
import re
import glob
from copy import deepcopy

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

class Joint:

    def __init__(self,tranformationMatrix,name,axis):
        self.name = name

        self.axis = np.zeros(3)
        self.axis[np.abs(axis)-1] = np.sign(axis)

        self.origin = tranformationMatrix[:3,-1]
        self.rpy = R.from_matrix(tranformationMatrix[:3,:3]).as_euler('xyz')

        self.tranformationMatrix = tranformationMatrix

    def __str__(self):
        return(self.name + " - Origin : " + str(self.origin) + " RPY : " + str(self.rpy) + " Axis : " + str(self.axis))

def ModeltoURDF(modelFilePath, robotName, show = False):

    ### BUILDING URDF 
    print("Building robot URDF !")

    #Reading model parameters
    with open(modelFilePath,"r") as file:
        modelParameters = yaml.load(file, Loader=yaml.FullLoader)

    TBase = np.reshape(np.array(modelParameters["robotBase"]),(4,4))
    TFlange = np.reshape(np.array(modelParameters["robotFlange"]),(4,4))
    piFull = modelParameters["robotParameters"]
    jointsAxis = modelParameters["jointsAxis"]

    def localT(parameters):
        T = np.eye(4)
        T[:3,-1] = parameters[1:4]
        T[:3,:3] = R.from_euler("xyz",parameters[4:7]).as_matrix()
        return(T)

    Joints = []

    ### STEP 1 : Robot base
    Joints.append(Joint(TBase,"joint0",jointsAxis[0]))

    ### STEP 2 : Robot links
    piFull = np.array(piFull)
    for i,parameters in enumerate(np.split(piFull,len(piFull)//7)[:-1]):
        deltaT = localT(parameters)
        Joints.append(Joint(deltaT,"joint" + str(i+1),jointsAxis[i+1]))      

    ### STEP 3 : Robot flange
    Joints.append(Joint(TFlange,"joint" + str(len(jointsAxis)-1),3))

    #Generate URDF file
    xacro = "echo $(xacro $(rospack find robot_arm_tools)/config/tools/none/" + robotName + "/" + robotName + "_none.urdf.xacro)"
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

    #Modify URDF file
    robot = URDF.load('/tmp/tmp.urdf')

    #Modified DH parameters convention
    #currentR = np.eye(4)
    #nextR = np.eye(4)

    globalTDH = np.eye(4)
    globalTModel = np.eye(4)
    linksNames = np.array([link.name for link in robot.links])

    if(len(robot.joints) == len(Joints)):
        robotJoints = robot.joints
        baseJoints = []
        flangeJoints = []
    else:
        robotJoints = []
        baseJoints = []
        flangeJoints = []
        for joint in robot.joints:
            if(joint not in robot.actuated_joints):
                if(len(robotJoints) == 0):
                    baseJoints.append(joint)
                elif(joint == robot.joints[-1]):
                    robotJoints.append(joint)
                else:
                    flangeJoints.append(joint)
            else:
                robotJoints.append(joint)

    for joint in baseJoints:
        globalTDH = np.dot(globalTDH,joint.origin)
        joint.origin = np.eye(4)

    for i,joint in enumerate(robotJoints):

        #Modified DH parameters convention
        #if(i == len(Joints) - 1):
        #    joint.origin = np.dot(np.linalg.inv(currentR),Joints[i].tranformationMatrix)
        #    joint.axis = Joints[i].axis
        #else:
        #    nextR[:3,:3] = np.dot(currentR,joint.origin)[:3,:3]
        #    joint.origin = np.dot(np.dot(np.linalg.inv(currentR),Joints[i].tranformationMatrix),nextR)
        #    joint.axis = np.dot(Joints[i].axis,nextR[:3,:3])
        #    currentR = deepcopy(nextR)

        #Model convention

        globalTDH = np.dot(globalTDH,joint.origin)
        joint.origin = Joints[i].tranformationMatrix
        joint.axis = Joints[i].axis
        globalTModel = np.dot(globalTModel,joint.origin)

        print(globalTDH)
        print(globalTModel)

        link = robot.links[np.where(linksNames == joint.child)[0][0]]
        
        #Inertial
        link.inertial.origin = np.dot(np.linalg.inv(globalTModel),np.dot(globalTDH,link.inertial.origin))

        #Visual
        for visual in link.visuals:
            visual.origin = np.dot(np.linalg.inv(globalTModel),np.dot(globalTDH,visual.origin))

        #Collision
        for collision in link.collisions:
            collision.origin = np.dot(np.linalg.inv(globalTModel),np.dot(globalTDH,collision.origin))

    for joint in flangeJoints:
        joint.origin = np.eye(4)

    #Save URDF file
    robot.save(os.getcwd() + "/" + robotName + ".urdf")

    if(show):
        import trimesh
        scene = trimesh.Scene()

        cfg = {}
        for joint in robot.actuated_joints:
            cfg[joint.name] = 0.0

        print(robot.link_fk(cfg=cfg)[robot.links[-1]])

        axis = trimesh.creation.axis(origin_size=0.01,axis_radius=0.01,axis_length=0.15,transform = robot.link_fk(cfg=cfg)[robot.links[-1]])
        scene.add_geometry(axis)

        meshes = robot.visual_trimesh_fk(cfg=cfg)

        mesh = trimesh.Trimesh()

        for submesh in meshes:
            pose = meshes[submesh]
            submesh.apply_transform(pose)
            mesh = trimesh.util.concatenate([mesh,submesh])

        newColors = np.array(mesh.visual.face_colors)
        newColors[:,-1] = 150
        mesh.visual = trimesh.visual.create_visual(face_colors=newColors,mesh=mesh)
        
        scene.add_geometry(mesh)

        scene.set_camera((np.pi/3,0.0,3*np.pi/4),3.5,(0,0,0.5))
        scene.show()

if __name__ == "__main__":
    try : 
        modelFilePath = sys.argv[1]
        robotName = sys.argv[2]
    except:
        modelFilePath = glob.glob("*.yaml")[0]
        currentParameters = np.array(re.split('/|_',os.getcwd()))
        currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
        robotName = currentParameters[0]
    
    ModeltoURDF(modelFilePath,robotName)