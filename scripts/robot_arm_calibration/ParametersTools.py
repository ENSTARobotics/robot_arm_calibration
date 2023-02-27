#!/usr/bin/python3

#System packages
from importlib import machinery
import sys
import os
import subprocess

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

def importParameters():
    """! Imports robot and marker paramters depending on how the file is called (run or import).
    """

    ### Retreive robot and marker names
    robotName = ""
    markerName = ""

    caller = sys._getframe(1)

    #Run case
    if caller.f_globals['__name__'] == "__main__":
        try:
            robotName = sys.argv[1]
            markerName = sys.argv[2]
        except:
            import re
            currentParameters = np.array(re.split('/|_',os.getcwd()))
            currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
            robotName = currentParameters[0]
            markerName = currentParameters[1]

    #Import case
    else:
        try:
            import re
            currentParameters = np.array(re.split('/|_',os.getcwd()))
            currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
            robotName = currentParameters[0]
            markerName = currentParameters[1]
        except:
            pass

    ### Get config directory path
    packagePath = "rospack find robot_arm_calibration"
    packageProcess = subprocess.run(packagePath,shell=True,stdout=subprocess.PIPE)
    packageOutput = str(packageProcess.stdout)[2:-3]
    configPath = packageOutput + "/config/nominalModels"

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