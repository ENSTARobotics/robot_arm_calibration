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
from robot_arm_calibration.URDFtoModel import URDFtoModel

class Joint:
    """! The Joint class.
    Defines an object describing a serial robot joint.
    """

    def __init__(self,tranformationMatrix,axis,offset):
        """! The Joint class initializer.
        @param transformationMatrix The 4x4 homogeneous local transformation matrix associated to the joint.
        @param axis                 The joint axis as a signed integer (1: X, 2: Y, 3: Z).
        @param offset               The joint offset.
        """

        #Joint axis vector
        self.axis = np.zeros(3)
        self.axis[np.abs(axis)-1] = np.sign(axis)

        #Joint rotation offset
        self.offset = offset

        #Joint transformation matrix (including offset)
        offsetTransformationMatrix = np.eye(4)
        offsetTransformationMatrix[:3,:3] = R.from_rotvec(self.axis*self.offset).as_matrix()
        self.tranformationMatrix = np.dot(tranformationMatrix,offsetTransformationMatrix)

def ModeltoURDF(modelFilePath, inputURDFFilePath, flangeLinkName = "", outputURDFFilePath = "", show = False):
    """! Modifies an URDF file according to a calibrated model file.

    @param modelFilePath        The path to the calibrated robot model.
    @param inputURDFFilePath    The path to the robot URDF or XACRO description file.
    @param flangeLinkName       The name of the robot's flange link.
    @param outputURDFFilePath   The path where the output URDF file will be saved - defaut = current_folder/inputURDFBaseName_output.urdf
    name of the folder where the output file will be saved - defaut = current_folder/robot_name/
    @param show                 Wether to visualize the 3D result or not - default = False.

    @return The generated URDF as a string.
    """

    ### Reading model parameters
    with open(modelFilePath,"r") as file:
        modelParameters = yaml.load(file, Loader=yaml.FullLoader)

    TBase = np.reshape(np.array(modelParameters["robotBase"]),(4,4))
    TFlange = np.reshape(np.array(modelParameters["robotFlange"]),(4,4))
    pi = modelParameters["robotParameters"]
    jointsAxis = modelParameters["jointsAxis"]

    return(ModeltoURDF_(TBase,TFlange,pi,jointsAxis,inputURDFFilePath,flangeLinkName,outputURDFFilePath,show))

def ModeltoURDF_(TBase, TFlange, robotParameters, jointsAxis, inputURDFFilePath, flangeLinkName = "", outputURDFFilePath = "", show = False):
    """! Modifies an URDF file according to a calibrated model file.

    @param TBase                The calibrated robot base transformation.
    @param TFlange              The calibrated robot flange transformation.
    @param robotParameters      The robot parameters vector.
    @param jointsAxis           The joints axis vector.
    @param inputURDFFilePath    The path to the robot URDF or XACRO description file.
    @param flangeLinkName       The name of the robot's flange link.
    @param outputURDFFilePath   The path where the output URDF file will be saved - defaut = current_folder/inputURDFBaseName_output.urdf
    @param show                 Wether to visualize the 3D result or not - default = False.

    @return The generated URDF as a string.
    """

    #Small change in names...
    pi = robotParameters

    ### Reading nominal model parameters
    JOINTS_NUMBER,JOINTS_AXIS,JOINTS_OFFSET,LINKS_TRANSLATIONS,LINKS_ROTATIONS,piN,TBaseN,TFlangeN = URDFtoModel(inputURDFFilePath,flangeLinkName)
    PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER+1)])

    ### Getting Joints 
    def localT(parameters, translations, rotations):
        T = np.eye(4)
        counter = 0

        for translation in translations:
            T[translation-1,-1] = parameters[counter]
            counter += 1

        angles = np.zeros(3)
        for rotation in rotations:
            angles[rotation-1] = parameters[counter]
            counter += 1
        T[:3,:3] = R.from_euler("xyz",angles).as_matrix()
        
        return(T)
    
    Joints = []
    JointsN = []

    #Remark : JOINTS_OFFSET always has the same shape, and should maybe be removed ?

    #Step 1 : Robot base joint
    Joints.append(Joint(TBase,jointsAxis[0],0))
    JointsN.append(Joint(TBaseN,JOINTS_AXIS[0],0))

    #Step 2 : Robot actuated joints
    for i in range(JOINTS_NUMBER - 1):
        if(JOINTS_OFFSET[i] == 0):
            deltaT = localT(pi[PARAMETERS_INDEX[i]:PARAMETERS_INDEX[i+1]],LINKS_TRANSLATIONS[i],LINKS_ROTATIONS[i])
            Joints.append(Joint(deltaT,jointsAxis[i+1],0))
            deltaTN = localT(piN[PARAMETERS_INDEX[i]:PARAMETERS_INDEX[i+1]],LINKS_TRANSLATIONS[i],LINKS_ROTATIONS[i])
            JointsN.append(Joint(deltaTN,JOINTS_AXIS[i+1],0))
        else:
            deltaT = localT(pi[PARAMETERS_INDEX[i]+1:PARAMETERS_INDEX[i+1]],LINKS_TRANSLATIONS[i],LINKS_ROTATIONS[i])
            Joints.append(Joint(deltaT,jointsAxis[i+1],pi[PARAMETERS_INDEX[i]]))
            deltaTN = localT(piN[PARAMETERS_INDEX[i]+1:PARAMETERS_INDEX[i+1]],LINKS_TRANSLATIONS[i],LINKS_ROTATIONS[i])
            JointsN.append(Joint(deltaTN,JOINTS_AXIS[i+1],piN[PARAMETERS_INDEX[i]]))

    #Step 3 : Robot flange joint
    Joints.append(Joint(TFlange,3,0))
    JointsN.append(Joint(TFlangeN,3,0))

    ### Load URDF file
    robot = getURDF(inputURDFFilePath)

    globalTRobot = np.eye(4)    #For the complete transformation without calibration and "native" frames conventions
    globalTCalibration = np.eye(4)  #For the complete transformation with calibration
    globalTN = np.eye(4)  #For the complete transformation without calibration and calibrated model frames conventions

    #Get URDF sorted joints, in the correct order
    baseJoints,actuatedJoints,flangeJoints = getSortedJoints(robot,flangeLinkName)

    #Adding base joints to the robot model
    for joint in baseJoints:
        globalTRobot = np.dot(globalTRobot,joint.origin)
        globalTCalibration = np.dot(globalTCalibration,joint.origin)
        globalTN = np.dot(globalTN,joint.origin)

    #Adding actuated joints to the robot model
    linksNames = np.array([link.name for link in robot.links])

    fixedJoints = np.array([joint for joint in robot.joints if joint.joint_type == "fixed"])
    fixedJointsParentNames = np.array([joint.parent for joint in robot.joints if joint.joint_type == "fixed"])

    for i,joint in enumerate(actuatedJoints):

        globalTRobot = np.dot(globalTRobot,joint.origin)

        if(i == 0):
            #Default : robot base joint is aligned on the reference world frame
            joint.origin = np.dot(np.linalg.inv(globalTCalibration),Joints[i].tranformationMatrix) 
            globalTN = JointsN[i].tranformationMatrix
        else:
            joint.origin = Joints[i].tranformationMatrix
            globalTN = np.dot(globalTN,JointsN[i].tranformationMatrix)

        joint.axis = Joints[i].axis

        globalTCalibration = np.dot(globalTCalibration,joint.origin)
        
        link = robot.links[np.where(linksNames == joint.child)[0][0]]
        
        #Inertial parameters
        link.inertial.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,link.inertial.origin))

        #Visual parameters
        for visual in link.visuals:  
            visual.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,visual.origin))      

        #Collision parameters
        for collision in link.collisions:
            collision.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,collision.origin))

        #Sometimes, and additional "fake" joint is added to the URDF file to handle collisions !
        linkFixedJoint = fixedJoints[np.where(fixedJointsParentNames == link.name)[0]]

        for fixedJoint in linkFixedJoint:

            #Not needed : it always an identity matrix
            #fixedJoint.origin = Joints[i].tranformationMatrix
            #fixedJoint.axis = Joints[i].axis

            fixedLink = robot.links[np.where(linksNames == fixedJoint.child)[0][0]]
        
            #Inertial parameters
            fixedLink.inertial.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,fixedLink.inertial.origin))

            #Visual parameters
            for visual in fixedLink.visuals:  
                visual.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,visual.origin))      

            #Collision parameters
            for collision in fixedLink.collisions:
                collision.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,collision.origin))


    #Adding flange joints to the robot model
    TFlangeCorrection = np.eye(4)
    for i,joint in enumerate(flangeJoints):

        globalTRobot = np.dot(globalTRobot,joint.origin)

        if(i == len(flangeJoints) - 1):
            globalTN = np.dot(globalTN,np.dot(np.linalg.inv(TFlangeCorrection),JointsN[-1].tranformationMatrix))
            joint.origin = np.dot(np.linalg.inv(TFlangeCorrection),Joints[-1].tranformationMatrix)
            joint.axis = Joints[-1].axis  
        else:
            globalTN = np.dot(globalTN,joint.origin)
            TFlangeCorrection = np.dot(TFlangeCorrection,joint.origin)    

        globalTCalibration = np.dot(globalTCalibration,joint.origin)
            
        link = robot.links[np.where(linksNames == joint.child)[0][0]]
        
        #Inertial parameters
        link.inertial.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,link.inertial.origin))

        #Visual parameters
        for visual in link.visuals:
            visual.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,visual.origin))

        #Collision parameters
        for collision in link.collisions:
            collision.origin = np.dot(np.linalg.inv(globalTN),np.dot(globalTRobot,collision.origin))

    ### Save URDF file
    if(outputURDFFilePath == ""):
        outputURDFFilePath = os.getcwd() + "/" + os.path.basename(inputURDFFilePath).split(".")[0] + "_output.urdf"

    robot.save(outputURDFFilePath)

    if(show):

        robot = URDF.load(outputURDFFilePath,True)

        import trimesh
        scene = trimesh.Scene()

        #Create display configuration
        cfg = {}
        for joint in robot.actuated_joints:
            cfg[joint.name] = (np.random.random() - 0.5)*np.pi

        #print(np.round(globalTN,3))
        #print(np.round(globalTRobot,3))
        #print(np.round(globalTCalibration,3))
        #print(np.round(robot.link_fk(cfg=cfg)[robot.links[-1]],3))

        #Add frames
        for link in robot.links:
            scene.add_geometry(trimesh.creation.axis(origin_size=0.01,axis_radius=0.01,axis_length=0.15,transform = robot.link_fk(cfg=cfg)[link]))

        #Handle robot visual meshes
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

    with open(outputURDFFilePath,"r") as f:
        output = f.read()
    return(output)

if __name__ == "__main__":
    try : 
        #TODO CATCH THESE ERRORS !
        modelFilePath = sys.argv[1]
        inputURDFFilePath = sys.argv[2]

        try:
            flangeLinkName = sys.argv[3]
        except:
            flangeLinkName = ""

        try:
            outputPath = sys.argv[4]
        except:
            outputPath = "/tmp/output.urdf"

    except:
        print("Invalid model and URDF/XACRO file paths !")
        sys.exit(-1)
    
    ModeltoURDF(modelFilePath,inputURDFFilePath,flangeLinkName,outputPath)
