#!/usr/bin/python3

#System packages
import sys
import os
import yaml

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_package_path():
    """! Returns robot_arm_calibration package path (for configuration files importation)

    @return         The package path.
    """

    import robot_arm_calibration
    module_path = os.path.dirname(robot_arm_calibration.__file__)
    scripts_path = os.path.dirname(module_path)
    package_path = os.path.dirname(scripts_path)
    return package_path

def import_robot_parameters_from_file(file_path, caller = None):
    """! Imports robot model parameters from a YAML file.

    @param file_path    The path to the YAML file containing the robot model parameters.
    @param caller       The caller frame #TODO REMOVE

    @return             The robot parameters dictionary.
    """

    input_parameters = None

    with open(file_path,'r') as file:
        input_parameters = yaml.load(file, Loader=yaml.FullLoader)

    if(input_parameters is None):
        raise ValueError("Empty file path for robot parameters importation !")

    output_parameters = {}

    output_parameters["JOINTS_AXIS"] = input_parameters["jointsAxis"]
    output_parameters["JOINTS_NUMBER"] = len(output_parameters["JOINTS_AXIS"])
    output_parameters["TBaseN"] = np.array(input_parameters["robotBase"]).reshape((4,4))
    output_parameters["TFlangeN"] = np.array(input_parameters["robotFlange"]).reshape((4,4))

    output_parameters["JOINTS_OFFSET"] = input_parameters["jointsOffset"]
    output_parameters["LINKS_TRANSLATIONS"] = input_parameters["linksTranslations"]
    output_parameters["LINKS_ROTATIONS"] = input_parameters["linksRotations"]

    output_parameters["piN"] = np.array(input_parameters["robot_parameters"])

    output_parameters["robotName"] = os.path.basename(file_path).split("_")[0]

    if(caller is None):
        caller = sys._getframe(1)

    caller.f_globals.update(output_parameters)
    return(output_parameters)


def import_robot_base_parameters_from_file(file_path, caller = None):
    """! Imports robot base transformation parameters from a YAML file.

    @param file_path    The path to the YAML file containing the robot base transformation parameters.
    @param caller       The caller frame #TODO REMOVE

    @return             The robot base transformation dictionary.
    """
        
    input_parameters = None

    try:
        with open(file_path,'r') as file:
            input_parameters = yaml.load(file, Loader=yaml.FullLoader)

        if(input_parameters is None):
            raise ValueError("Invalid file path for robot base transformation importation !")
    except (FileNotFoundError, ValueError):
        input_parameters = {}

    output_parameters = {}

    if("robotBaseOffset" in input_parameters.keys()):
        output_parameters["TBaseOffset"] = np.array(input_parameters["robotBaseOffset"]).reshape((4,4))
    else:
        output_parameters["TBaseOffset"] = np.eye(4)

    if(caller is None):
        caller = sys._getframe(1)

    caller.f_globals.update(output_parameters)
    return(output_parameters)

def import_tools_parameters_from_file(file_path, caller = None):
    """! Imports tool model parameters from a YAML file.

    @param file_path    The path to the YAML file containing the tool model parameters.
    @param caller       The caller frame #TODO REMOVE

    @return             The tool parameters dictionary.
    """
        
    input_parameters = None

    try:
        with open(file_path,'r') as file:
            input_parameters = yaml.load(file, Loader=yaml.FullLoader)

        if(input_parameters is None):
            raise ValueError("Invalid file path for tools parameters importation !")
    except (FileNotFoundError, ValueError):
        input_parameters = {}

    output_parameters = {}

    if("robotTools" in input_parameters.keys()):
        output_parameters["TToolN"] = np.array(input_parameters["robotTools"]).reshape((-1,4,4))
        output_parameters["N_EEPOINTS"] = len(output_parameters["TToolN"])
        if(output_parameters["N_EEPOINTS"] > 1):
            output_parameters["EEItemsNames"] = ["EEPosition" + str(i+1) for i in range(output_parameters["N_EEPOINTS"])]
        else:
            output_parameters["EEItemsNames"] = ["EEPose"]
        if("additionnalParameters" in input_parameters.keys()):
            output_parameters["EEItemsNames"].append(input_parameters["additionnalParameters"])
    else:
        print("Invalid file path for tools parameters importation, defaulting to no tool.")
        output_parameters["TToolN"] = np.array([np.eye(4)])
        output_parameters["N_EEPOINTS"] = len(output_parameters["TToolN"])
        if(output_parameters["N_EEPOINTS"] > 1):
            output_parameters["EEItemsNames"] = ["EEPosition" + str(i+1) for i in range(output_parameters["N_EEPOINTS"])]
        else:
            output_parameters["EEItemsNames"] = ["EEPose"]

    output_parameters["toolName"] = os.path.basename(file_path).split("_")[0]

    if(caller is None):
        caller = sys._getframe(1)

    caller.f_globals.update(output_parameters)
    return(output_parameters)

def import_all_parameters_from_file(file_path, caller = None):
    """! Imports all model parameters from a YAML file.

    @param file_path    The path to the YAML file containing all model parameters.
    @param caller       The caller frame #TODO REMOVE

    @return             The model parameters dictionary.
    """
    
    if(caller is None):
        caller = sys._getframe(1)

    robot_parameters = import_robot_parameters_from_file(file_path, caller)
    base_parameters = import_robot_base_parameters_from_file(file_path, caller)
    tool_parameters = import_tools_parameters_from_file(file_path, caller)

    all_parameters = {**robot_parameters,**base_parameters,**tool_parameters}

    #Extract robot and tool names from folder structure
    if("calibratedModels" in file_path):
        split_name = np.array(file_path.split("/"))
        split_name = split_name[np.where(split_name == "calibratedModels")[0][0] + 1:]
        split_name = split_name[0].split("_")
        robot_name = split_name[0]
        tool_name = split_name[1]
    #Extract robot and tool names from file name
    else:
        split_name = os.path.basename(os.path.dirname(file_path)).split("_")
        robot_name = split_name[0]
        tool_name = "" if len(split_name) == 1 else split_name[1]

    all_parameters["robotName"] = robot_name
    all_parameters["toolName"] = tool_name
    caller.f_globals.update({"robotName":robot_name,"toolName":tool_name})
    return(all_parameters)

def import_all_parameters_from_names(robot_name = "", tool_name = ""):
    """! Imports all model parameters from robot and tool names

    @param robot_name    The name of the robot.
    @param tool_name     The name of the tool.

    @return             The model parameters dictionary.
    """

    caller = sys._getframe(1)
    model_path = get_package_path() + "/config/nominalModels/"

    #Import parameters using the provided names
    try:
        robot_parameters = import_robot_parameters_from_file(model_path +  "robots/" + robot_name + "/" + robot_name + "_parameters.yaml", caller)
        base_parameters = import_robot_base_parameters_from_file(model_path +  "robots/" + robot_name + "/" + robot_name + "_base_parameters.yaml", caller)
        tool_parameters = import_tools_parameters_from_file(model_path + "tools/" + tool_name + "/" + tool_name + "_parameters.yaml", caller)

    #Retrieve robot and tool names from folder structure
    except (FileNotFoundError, ValueError):
        current_parameters = np.array(os.getcwd().split("/"))
        current_parameters = current_parameters[np.where(current_parameters == "calibratedModels")[0][0] + 1:]
        current_parameters = current_parameters[0].split("_")

        #Assumption : the robot name does not contain "_", TODO better : neither the marker name, nor the robot name contains "_"
        robot_name = current_parameters[0]
        tool_name = "_".join(current_parameters[1:])  

        robot_parameters = import_robot_parameters_from_file(model_path +  "robots/" + robot_name + "/" + robot_name + "_parameters.yaml", caller)
        base_parameters = import_robot_base_parameters_from_file(model_path +  "robots/" + robot_name + "/" + robot_name + "_base_parameters.yaml", caller)
        tool_parameters = import_tools_parameters_from_file(model_path + "tools/" + tool_name + "/" + tool_name + "_parameters.yaml", caller)

    all_parameters = {**robot_parameters,**base_parameters,**tool_parameters}
    all_parameters["robotName"] = robot_name
    all_parameters["toolName"] = tool_name
    caller.f_globals.update({"robotName":robot_name,"toolName":tool_name})
    return(all_parameters)

#TODO Sort arguments properly !
def save_parameters(file_path, transformation_base_first_joint = None, transformation_last_joint_flange = None, pi = None, transformation_flange_tool = None, transformation_world_base = None):
    """! Save all model parameters in a YAML file.

    @param file_path                        The path where to save the model parameters.
    @param transformation_base_first_joint  The robot base to first joint transformation matrix.
    @param transformation_last_joint_flange The robot last joint to flange transformation matrix.
    @param pi                               The robot kinematic parameters vector.
    @param transformation_flange_tool       The robot flange to tool(s) transformation matrix.
    @param transformation_world_base        The world to robot base transformation matrix.
    """

    #Robot model
    output_parameters = {}

    #Robot base offset
    if(transformation_world_base is not None):
        output_parameters["robotBaseOffset"] = transformation_world_base.flatten().tolist()

    #Robot tool(s)
    if(transformation_flange_tool is not None):
        output_parameters["robotTools"] = transformation_flange_tool.flatten().tolist()

    #Robot parameters
    if(transformation_base_first_joint is not None):
        output_parameters["robotBase"] = transformation_base_first_joint.flatten().tolist()
    if(transformation_last_joint_flange is not None):
        output_parameters["robotFlange"] = transformation_last_joint_flange.flatten().tolist()
    if(pi is not None):
        output_parameters["robot_parameters"] = pi.tolist()

        #TODO REMOVE !
        caller = sys._getframe(1)
        output_parameters["jointsAxis"] = caller.f_globals.get("JOINTS_AXIS")
        output_parameters["jointsOffset"] = caller.f_globals.get("JOINTS_OFFSET")
        output_parameters["linksTranslations"] = caller.f_globals.get("LINKS_TRANSLATIONS")
        output_parameters["linksRotations"] = caller.f_globals.get("LINKS_ROTATIONS")

    if(not os.path.isdir(os.path.dirname(file_path))):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path,'w') as file:
        yaml.dump(output_parameters, file)

def get_identifiable_parameters(transformation_base_first_joint = None, transformation_last_joint_flange = None, pi = None, transformation_flange_tool = None, transformation_world_base = None):
    """! Converts readable parameters into an identifiable parameters vector.

    @param transformation_base_first_joint  The robot base to first joint transformation matrix.
    @param transformation_last_joint_flange The robot last joint to flange transformation matrix.
    @param pi                               The robot kinematic parameters vector.
    @param transformation_flange_tool       The robot flange to tool(s) transformation matrix.
    @param transformation_world_base        The world to robot base transformation matrix.

    @return                                 The identifiable parameters vector.
    """

    ### Robot base ###
    
    transformation_world_first_joint = np.dot(transformation_world_base,transformation_base_first_joint)
    P_world_first_joint = transformation_world_first_joint[:3,-1]
    R_world_first_joint = R.from_matrix(transformation_world_first_joint[:3,:3])

    ### Robot tool(s) ###

    #Partial pose measurements
    if(len(transformation_flange_tool.shape) > 1):
        list_transformation_last_joint_tool = np.empty((len(transformation_flange_tool.shape),4,4))
        for i in range(len(transformation_flange_tool.shape)):
            list_transformation_last_joint_tool[i] = np.dot(transformation_last_joint_flange,transformation_flange_tool[i])
        P_last_joint_tool = list_transformation_last_joint_tool[:,:3,-1]

        parameters = np.hstack((P_world_first_joint,R.as_mrp(R_world_first_joint),pi,P_last_joint_tool.flatten()))

    #Full pose measurements
    else:
        transformation_last_joint_tool = np.dot(transformation_last_joint_flange,transformation_flange_tool[0])
        P_last_joint_tool = transformation_last_joint_tool[:3,-1]
        R_last_joint_tool = R.from_matrix(transformation_last_joint_tool[:3,:3])

        parameters = np.hstack((P_world_first_joint,R.as_mrp(R_world_first_joint),pi,P_last_joint_tool,R.as_mrp(R_last_joint_tool)))
    
    return(parameters)
