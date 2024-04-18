#!/usr/bin/python3

#System packages
import subprocess
import re

#Utility packages
import numpy as np

#URDF package
from urchin import URDF

#ROS package
import rospkg

def clean_xacro_output(output):
    """! Cleans the raw XACRO output by parsing unrecognized "package://" and "file://" tags.

    @param output   The raw XACRO output.

    @return         The cleaned XACRO output.
    """

    rospack = rospkg.RosPack()
    packages = np.unique(np.array([output[int(m.start()):int(m.end())] for m in re.finditer('package://[a-zA-Z_]+/', output)]))

    for package in packages:
        output = output.replace(package,rospack.get_path(package.split("/")[-2]) + "/")

    output = output.replace("file://","")
    return(output)

def get_urdf(file_path, **kwargs):
    """! Extract the URDF model contained in an URDF or a XACRO file.

    @param file_path    The path to the URDF/XACRO file.
    @param **kwargs     The eventual arguments to pass to the xacro command while parsing the XACRO file.

    @return             The URDF model extracted using urchin.
    """

    #XACRO file
    if(file_path.split(".")[-1] == "xacro"):

        #Execute XACRO command and get 
        xacro = "echo $(xacro " + file_path
        for key, value in kwargs.items():
            xacro += " " + key + ":=" + str(value)
        xacro += ")" 
        xacro_process = subprocess.run(xacro,shell=True,stdout=subprocess.PIPE)
        output = clean_xacro_output(str(xacro_process.stdout)[2:-3])

        with open("/tmp/tmp.urdf", "w") as f:
            f.write(output)

        ### Modify URDF file
        robot = URDF.load('/tmp/tmp.urdf',True)

    #URDF file
    else:
        robot = URDF.load(file_path,True)

    return(robot)

def get_distance_to_base_link(robot, link):
    """! Computes the distance (as number of "real" joints) between a given link and the robot base link

    @param robot    The urchin robot URDF model.
    @param link     The considered link.

    @return         The distance.
    """

    links_names = np.array([link.name for link in robot.links])
    joints_child_name = np.array([joint.child for joint in robot.joints])

    tmp_link = link
    distance = 0
    while(tmp_link != robot.base_link):
        #Update current joint
        tmp_joint = robot.joints[np.where(joints_child_name == tmp_link.name)[0][0]]
        #Update current link
        new_link = robot.links[np.where(links_names == tmp_joint.parent)[0][0]]

        #Remove "fake" joints (e.g. for coarse self collisions "sc")
        if(new_link.name not in tmp_link.name):
            distance+=1
        tmp_link = new_link
    return(distance)

def get_sorted_joints(robot,flange_link_name):
    """! Stores and sorts the joints linking the robot flange to the robot base.

    @param robot                The urchin robot URDF model.
    @param flange_link_name     The name of the robot flange link.

    @return                     The robot base joints.
    @return                     The robot actuated joints.
    @return                     The robot flange joints.
    """

    links_names = np.array([link.name for link in robot.links])
    
    #Find a correct flange link name to proceed
    if(flange_link_name not in links_names or flange_link_name == ""):
        flange_link_name = robot.end_links[np.argmax([get_distance_to_base_link(robot,link) for link in robot.end_links])].name    
        print("Invalid flange link name, defaulting to : " + flange_link_name)

    flange_link = robot.links[np.where(links_names == flange_link_name)[0][0]]

    base_joints = []
    acturated_joints = []
    flange_joints = []

    #Start from the flange link, go back to the base link and sort URDF joints
    tmp_link = flange_link
    joints_child_names = np.array([joint.child for joint in robot.joints])

    while(tmp_link != robot.base_link):
        #Update current joint
        tmp_joint = robot.joints[np.where(joints_child_names == tmp_link.name)[0][0]]

        #Investigating not actuated intermediate joints : 
        if(tmp_joint not in robot.actuated_joints):
            #Conventionnal base joints (e.g. "base_inertia")
            if(len(acturated_joints) != 0):
                base_joints.append(tmp_joint)
            #Conventionnal flange joints (e.g. "flange")
            else:
                flange_joints.append(tmp_joint)
        else:
            acturated_joints.append(tmp_joint)

        #Update current link
        tmp_link = robot.links[np.where(links_names == tmp_joint.parent)[0][0]]

    #Put everything in the correct order and return
    return(base_joints[::-1],acturated_joints[::-1],flange_joints[::-1])
