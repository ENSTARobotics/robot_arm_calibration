#!/usr/bin/python3

#System packages
from importlib import machinery
import sys
import os

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

#ROS package
import rospkg
rospack = rospkg.RosPack()

def importParameters(robotName = "", markerName = ""):
    """! Imports robot and marker paramters
    """

    ### Retreive robot and marker names
    caller = sys._getframe(1)
    if(robotName == "" and markerName == ""):
        try:
            import re
            currentParameters = np.array(re.split('/|_',os.getcwd()))
            currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
            robotName = currentParameters[0]
            markerName = currentParameters[1]
        except:
            print("Invalid robot name for robot parameters importation ! Are you in a proper calibrated model folder or have you specified a correct robot name ? ")
            exit(1)

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
    except:
        print("Invalid robot name for robot parameters importation ! Are you in a proper calibrated model folder or have you specified a correct robot name ? ")
        exit(1)

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
        print("Invalid robot name for robot parameters importation ! Are you in a proper calibrated model folder or have you specified a correct marker name ? ")
        exit(1)

    caller.f_globals.update({"robotName":robotName,"markerName":markerName})

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

    parameters = np.hstack((PBase,R.as_euler(RBase,"xyz"),pi,PTool.flatten()))

    return(parameters)