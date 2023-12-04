#!/usr/bin/python3

#System packages
from importlib import machinery
import sys
import os
import yaml

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

#ROS package
import rospkg
rospack = rospkg.RosPack()

def importParametersFromFile(filePath):
    """! Imports robot and marker parameters from a yaml file.
    """

    calibratedModel = None

    try:
        with open(filePath,'r') as file:
            calibratedModel = yaml.load(file, Loader=yaml.FullLoader)

        if(calibratedModel is None):
            raise ValueError
    except:
        print("Invalid file path for robot parameters importation !")
        sys.exit(-1)

    caller = sys._getframe(1)
    robotModelData = {}

    robotModelData["JOINTS_AXIS"] = np.array(calibratedModel["jointsAxis"])
    robotModelData["JOINTS_NUMBER"] = len(robotModelData["JOINTS_AXIS"])
    robotModelData["TBaseN"] = np.array(calibratedModel["robotBase"]).reshape((4,4))
    robotModelData["TFlangeN"] = np.array(calibratedModel["robotFlange"]).reshape((4,4)), 
    pi0Full = np.array(calibratedModel["robotParameters"])

    robotModelData["JOINTS_OFFSET"] = []
    robotModelData["LINKS_TRANSLATIONS"] = []
    robotModelData["LINKS_ROTATIONS"] = []

    for pi0Local in np.split(pi0Full, robotModelData["JOINTS_NUMBER"]):
        if(pi0Local[0] == 0):
            robotModelData["JOINTS_OFFSET"].append(0)
        else:
            robotModelData["JOINTS_OFFSET"].append(1)

        local_link_translations = []
        for i,link_translation in enumerate(pi0Local[1:4]):
            if(link_translation != 0):
                local_link_translations.append(i+1)
        robotModelData["LINKS_TRANSLATIONS"].append(local_link_translations)

        local_link_rotations = []
        for i,link_rotation in enumerate(pi0Local[4:7]):
            if(link_rotation != 0):
                local_link_rotations.append(i+1)
        robotModelData["LINKS_ROTATIONS"].append(local_link_rotations)

    robotModelData["pi0"] = pi0Full[np.where(pi0Full != 0)]

    caller.f_globals.update(robotModelData)

    #TODO : Case where a Tool is defined !
    caller.f_globals.update({"TToolN_marker": np.array([np.eye(4)])})
    caller.f_globals.update({"N_EEPOINTS": 1})
    caller.f_globals.update({"EEPointsItemsNames": ["EEPosition"]})

def importParameters(robotName = "", markerName = ""):
    """! Imports robot and marker parameters from the nominal .py file - TO DEPRECATE !
    => Nominal models should be stored as a yaml file, and imported with importParametersFromFile() function.
    """

    ### Retreive robot and marker names
    caller = sys._getframe(1)

    # "Raw" robot model
    if(robotName != "" and markerName == ""):
        caller.f_globals.update({"TToolN_marker": np.array([np.eye(4)])})
        caller.f_globals.update({"N_EEPOINTS": 1})
        caller.f_globals.update({"EEPointsItemsNames": ["EEPosition"]})

    elif(robotName == "" and markerName == ""):
        try:
            import re
            currentParameters = np.array(os.getcwd().split("/"))
            currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
            currentParameters = currentParameters[0].split("_")
            #Assumption : the robot name does not contain "_", TODO better : neither the marker name, nor the robot name contains "_"
            robotName = currentParameters[0]
            markerName = "_".join(currentParameters[1:])

        except:
            try:
                robotName = caller.f_globals.get("robotName")
                markerName = caller.f_globals.get("markerName")
            except:
                print("Invalid robot name for robot parameters importation ! Are you in a proper calibrated model folder or have you specified a correct robot name ? ")
                sys.exit(-1)

    ### Get config directory path
    configPath = rospack.get_path("robot_arm_calibration") + "/config/nominalModels"

    ### Import robot parameters from dedicated file
    configPathRobot = configPath + "/" + robotName + "/"

    try:
        module = machinery.SourceFileLoader(robotName + "_parameters",configPathRobot + robotName + "_parameters.py").load_module()

        if "__all__" in module.__dict__:
            names = module.__dict__["__all__"]
        else:
            names = [x for x in module.__dict__ if not x.startswith("_")]

        caller.f_globals.update({k: getattr(module, k) for k in names})
        caller.f_globals.update({"robotName":robotName,"configPathRobot":configPathRobot})
    except:
        print("Invalid robot name for robot parameters importation ! Are you in a proper calibrated model folder or have you specified a correct robot name ? ")
        sys.exit(-1)

    if(markerName != ""):
        ### Import marker parameters from dedicated file
        configPathMarker = configPath + "/markers/"
        try:
            module = machinery.SourceFileLoader(markerName + "_parameters",configPathMarker + markerName + "_parameters.py").load_module()

            if "__all__" in module.__dict__:
                names = module.__dict__["__all__"]
            else:
                names = [x for x in module.__dict__ if not x.startswith("_")]

            caller.f_globals.update({k: getattr(module, k) for k in names})
        except:
            print("Invalid marker name for robot parameters importation ! Are you in a proper calibrated model folder or have you specified a correct marker name ? ")
            sys.exit(-1)

        caller.f_globals.update({"configPathMarker":configPathMarker})

    caller.f_globals.update({"markerName":markerName})

def getIdentificationParameters(TBase,TFlange,TTool_marker,pi):
    """! Converts readable parameters into an identifiable parameters vector.

    @param TBase            The robot base homogeneous transformation matrix (relative to the measuring device reference frame).
    @param TFLange          The robot flange homogeneous transformation matrix (relative to the last robot joint frame).
    @param TTool_marker     The list containing the end effector points homogeneous transformation matrix (relative to the robot flange).
    @param pi               The robot kinematic parmeters vector.

    @return     The identifiable parameters vector.
    """

    TTool = np.empty((len(TTool_marker),4,4))
    for j in range(len(TTool_marker)):
        TTool[j] = np.dot(TFlange,TTool_marker[j])

    PBase = TBase[:3,-1]
    RBase = R.from_matrix(TBase[:3,:3])
    PTool = TTool[:,:3,-1]

    parameters = np.hstack((PBase,R.as_rotvec(RBase),pi,PTool.flatten()))

    return(parameters)