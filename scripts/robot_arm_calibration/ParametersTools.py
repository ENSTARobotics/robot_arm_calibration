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

def importRobotParametersFromFile(filePath, caller = None):
    inputParameters = None

    try:
        with open(filePath,'r') as file:
            inputParameters = yaml.load(file, Loader=yaml.FullLoader)

        if(inputParameters is None):
            raise ValueError
    except:
        print("Invalid file path for robot parameters importation !")
        sys.exit(-1)

    outputParameters = {}

    outputParameters["JOINTS_AXIS"] = inputParameters["jointsAxis"]
    outputParameters["JOINTS_NUMBER"] = len(outputParameters["JOINTS_AXIS"])
    outputParameters["TBaseN"] = np.array(inputParameters["robotBase"]).reshape((4,4))
    outputParameters["TFlangeN"] = np.array(inputParameters["robotFlange"]).reshape((4,4))

    outputParameters["JOINTS_OFFSET"] = inputParameters["jointsOffset"]
    outputParameters["LINKS_TRANSLATIONS"] = inputParameters["linksTranslations"]
    outputParameters["LINKS_ROTATIONS"] = inputParameters["linksRotations"]

    outputParameters["piN"] = np.array(inputParameters["robotParameters"])

    if(caller is None):
        caller = sys._getframe(1)

    caller.f_globals.update(outputParameters)
    caller.f_globals.update({"robotName":os.path.basename(filePath).split("_")[0]})

def importToolParametersFromFile(filePath, caller = None):
    inputParameters = None

    try:
        with open(filePath,'r') as file:
            inputParameters = yaml.load(file, Loader=yaml.FullLoader)

        if(inputParameters is None):
            raise ValueError
    except:
        inputParameters = {}

    outputParameters = {}

    if("robotTools" in inputParameters.keys()):
        outputParameters["TToolN"] = np.array(inputParameters["robotTools"]).reshape((-1,4,4))
        outputParameters["N_EEPOINTS"] = len(outputParameters["TToolN"])
        outputParameters["EEItemsNames"] = ["EEPosition" + str(i+1) for i in range(outputParameters["N_EEPOINTS"])]
    else:
        print("Invalid file path for tools parameters importation, defaulting to no tool.")
        outputParameters["TToolN"] = np.array([np.eye(4)])
        outputParameters["N_EEPOINTS"] = len(outputParameters["TToolN"])
        outputParameters["EEItemsNames"] = ["EEPosition" + str(i+1) for i in range(outputParameters["N_EEPOINTS"])]

    if(caller is None):
        caller = sys._getframe(1)

    caller.f_globals.update(outputParameters)
    caller.f_globals.update({"toolName":os.path.basename(filePath).split("_")[0]})

def importAllParametersFromFile(filePath, caller = None):
    
    if(caller is None):
        caller = sys._getframe(1)

    importRobotParametersFromFile(filePath, caller)
    importToolParametersFromFile(filePath, caller)

    splitName = os.path.basename(os.path.dirname(filePath)).split("_")
    robotName = splitName[0]
    toolName = "" if len(splitName) == 1 else splitName[1]

    caller.f_globals.update({"robotName":robotName,"toolName":toolName})

def importParameters(robotName = "", toolName = ""):

    caller = sys._getframe(1)
    import rospkg
    rospack = rospkg.RosPack()
    configFolderPath = rospack.get_path("robot_arm_calibration") + "/config/nominalModels/"

    try:
        importRobotParametersFromFile(configFolderPath +  "robots/" + robotName + "/" + robotName + "_parameters.yaml", caller)
        importToolParametersFromFile(configFolderPath + "tools/" + toolName + "/" + toolName + "_parameters.yaml", caller)

    except:
        import os
        currentParameters = np.array(os.getcwd().split("/"))
        currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
        currentParameters = currentParameters[0].split("_")

        #Assumption : the robot name does not contain "_", TODO better : neither the marker name, nor the robot name contains "_"
        robotName = currentParameters[0]
        toolName = "_".join(currentParameters[1:])  

        importRobotParametersFromFile(configFolderPath +  "robots/" + robotName + "/" + robotName + "_parameters.yaml", caller)
        importToolParametersFromFile(configFolderPath + "tools/" + toolName + "/" + toolName + "_parameters.yaml", caller)

    caller.f_globals.update({"robotName":robotName,"toolName":toolName}) 

def saveParameters(filePath, TBase = None, TFlange = None, pi = None, TTool = None):

    #Robot model
    outputParameters = {}

    #Robot tools
    if(TTool is not None):
        outputParameters["robotTools"] = TTool.flatten().tolist()

    #Robot parameters
    if(TBase is not None):
        outputParameters["robotBase"] = TBase.flatten().tolist()
    if(TFlange is not None):
        outputParameters["robotFlange"] = TFlange.flatten().tolist()
    if(pi is not None):
        outputParameters["robotParameters"] = pi.tolist()
        caller = sys._getframe(1)
        outputParameters["jointsAxis"] = caller.f_globals.get("JOINTS_AXIS")
        outputParameters["jointsOffset"] = caller.f_globals.get("JOINTS_OFFSET")
        outputParameters["linksTranslations"] = caller.f_globals.get("LINKS_TRANSLATIONS")
        outputParameters["linksRotations"] = caller.f_globals.get("LINKS_ROTATIONS")

    if(not os.path.isdir(os.path.dirname(filePath))):
        os.makedirs(os.path.dirname(filePath))
    with open(filePath,'w') as file:
        yaml.dump(outputParameters, file)

def getIdentificationParameters(TBase,TFlange,TTool,pi):
    """! Converts readable parameters into an identifiable parameters vector.

    @param TBase            The robot base homogeneous transformation matrix (relative to the measuring device reference frame).
    @param TFLange          The robot flange homogeneous transformation matrix (relative to the last robot joint frame).
    @param TTool            The list containing the end effector points homogeneous transformation matrix (relative to the robot flange).
    @param pi               The robot kinematic parmeters vector.

    @return     The identifiable parameters vector.
    """

    TFlangeTool = np.empty((len(TTool),4,4))
    for j in range(len(TTool)):
        TFlangeTool[j] = np.dot(TFlange,TTool[j])

    PBase = TBase[:3,-1]
    RBase = R.from_matrix(TBase[:3,:3])
    PFlangeTool = TFlangeTool[:,:3,-1]

    parameters = np.hstack((PBase,R.as_rotvec(RBase),pi,PFlangeTool.flatten()))

    return(parameters)