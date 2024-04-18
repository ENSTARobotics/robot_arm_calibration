#!/usr/bin/python3

#System packages
import cloup
import functools

#Utility packages
#from jax import jit    
#Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once. Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.
import jax.numpy as np
from jax.scipy.spatial.transform import Rotation as R

#Jacobian computation package
from jax import jacfwd, jacrev  
#jacfwd uses forward-mode automatic differentiation, which is more efficient for “tall” Jacobian matrices, while jacrev uses reverse-mode, which is more efficient for “wide” Jacobian matrices. For matrices that are near-square, jacfwd probably has an edge over jacrev.   

#Geometric parameters tools
from robot_arm_calibration.ParametersTools import importAllParametersFromNames, importAllParametersFromFile

def load_geometric_model(func):
    @cloup.option("--robot_name", type=str)
    @cloup.option("--tool_name", type=str, default="")
    @cloup.option("--parameters_file", type=str)
    @cloup.constraint(cloup.constraints.RequireExactly(1),["robot_name", "parameters_file"])
    @functools.wraps(func)
    def wrapper(robot_name, tool_name, parameters_file, *args, **kwargs):
        parameters = {}
        if parameters_file:
            parameters = importAllParametersFromFile(parameters_file)
        else:
            parameters = importAllParametersFromNames(robot_name, tool_name)

        return func(robotModel = RobotModel(**parameters), *args, **kwargs)
    return wrapper

class RobotModel:

    def __init__(self, **kwargs):
        """! Initializes the robot geometric model."""

        ### ROBOT ###
        self.robot_name = kwargs["robotName"]

        # Model parameters layout
        self.joints_number = kwargs["JOINTS_NUMBER"]
        self.joints_axis = kwargs["JOINTS_AXIS"]
        self.joints_offset = kwargs["JOINTS_OFFSET"]
        self.links_translations = kwargs["LINKS_TRANSLATIONS"]
        self.links_rotations = kwargs["LINKS_ROTATIONS"]
        self.parameters_index = np.array([sum(self.joints_offset[:i]) + 
                                          sum(map(len,self.links_translations[:i])) + 
                                          sum(map(len,self.links_rotations[:i])) 
                                          for i in range(self.joints_number+1)])
        
        # Model nominal parameters
        self.transformation_base_N = kwargs["TBaseN"]
        self.transformation_flange_N = kwargs["TFlangeN"]
        self.pi_N = kwargs["piN"]

        ### TOOL ###  
        self.tool_name = kwargs["toolName"]
        self.tool_points_number = kwargs["N_EEPOINTS"]
        self.tool_points_names = kwargs["EEItemsNames"]

        # Model nominal parameters
        self.transformation_flange_tool_N = kwargs["TFlangeN"]

        ### BASE OFFSET ###
        self.transformation_world_base = kwargs["TBaseOffset"]

        ### MODEL FUNCTION ###
        if(self.tool_points_number > 1):
            self.P_func = self.P_partial_func
        else:
            self.P_func = self.P_full_func

        self.J_func = jacfwd(self.P_func)

    def Ti(self, i, q, pi = None):
        """! Computes the local homogeneous transformation matrix for the ith robot joint.

        @param i   The robot joint index.
        @param q   The robot angular configuration (i.e. actuators values).
        @param pi  The robot kinematic parameters vector.

        @return  The local homogeneous transformation matrix, the list of the involved successive primitive transformations (single axis translation/rotation) matrices.
        """

        if pi is None:
            pi = self.pi_N

        T = np.eye(4)

        parameters = pi[self.parameters_index[i]:self.parameters_index[i+1]]

        ###Joint rotation and offset
        Ti = np.eye(4)
        angles = np.zeros(3)

        angles = angles.at[np.abs(self.joints_axis[i])-1].set(np.sign(self.joints_axis[i])*q[i])
        Ti = Ti.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())
        T = np.dot(T,Ti)

        if(self.joints_offset[i] == 1):
            Ti = np.eye(4)
            angles = np.zeros(3)

            angles = angles.at[np.abs(self.joints_axis[i])-1].set(parameters[0])
            Ti = Ti.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())
            T = np.dot(T,Ti)

        ###Link translation
        for n,axis in enumerate(self.links_translations[i]):
            Ti = np.eye(4)
            Ti = Ti.at[np.abs(axis)-1,-1].set(np.sign(axis)*parameters[self.joints_offset[i] + n])

            T = np.dot(T,Ti)

        ###Link rotation
        for n,axis in enumerate(self.links_rotations[i]):
            Ti = np.eye(4)
            angles = np.zeros(3)

            angles = angles.at[np.abs(axis)-1].set(np.sign(axis)*parameters[self.joints_offset[i] + len(self.links_translations[i]) + n])
            Ti = Ti.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())

            T = np.dot(T,Ti)

        return(T)

    def T(self, q, pi = None, transformation_world_base = None, transformation_flange_tool = None):
        """! Computes the global homogeneous transformation matrix between the world reference frame and the robot tool.

        @param q        The robot angular configuration (i.e. actuators values).
        @param pi       The robot kinematic parameters vector.
        @param transformation_world_base    The homogeneous transformation matrix between the world reference frame and the robot base.
        @param transformation_flange_tool    The homogeneous transformation matrix between the last robot joint and the robot tool.

        @return  The global homogeneous transformation matrix.
        """

        if pi is None:
            pi = self.pi_N
        if transformation_world_base is None:
            transformation_world_base = self.transformation_base_N
        if transformation_flange_tool is None:
            transformation_flange_tool = self.transformation_flange_N

        T = transformation_world_base
        for i in range(self.joints_number):
            T = np.dot(T,self.Ti(i,q,pi))
        return(np.dot(T,transformation_flange_tool))
    
    def P(self, q, pi = None, transformation_world_base = None, transformation_flange_tool = None):
        """! Computes the position of end effector points in the world reference frame.

        @param q        The robot angular configuration (i.e. actuators values).
        @param pi       The robot kinematic parameters vector.
        @param transformation_world_base    The homogeneous transformation matrix between the world reference frame and the robot base frame.
        @param transformation_flange_tool    The homogeneous transformation matrix between the last robot joint and the robot tool.

        @return  The positions of end effector points
        """

        if pi is None:
            pi = self.pi_N
        if transformation_world_base is None:
            transformation_world_base = self.transformation_base_N
        if transformation_flange_tool is None:
            transformation_flange_tool = self.transformation_flange_N

        T = transformation_world_base
        for i in range(self.joints_number):
            T = np.dot(T,self.Ti(i,q,pi))
        P = np.vstack([np.dot(T,t)[:3,-1] for t in transformation_flange_tool])
        return(P)
    
    def T_func(self, parameters, q):
        """! Computes the global homogeneous transformation matrix between the robot base frame and the robot flange frame.

        @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
        @param q            The robot angular configuration (i.e. actuators values).

        @return  The global homogeneous transformation matrix between the robot base frame and the robot flange frame.
        """
        
        pi = parameters[6:6+len(self.pi_N)]
        T = np.eye(4)
        for i in range(self.joints_number):
            T = np.dot(T,self.Ti(i,q,pi))
        return(T)
    
    def P_partial_func(self, parameters, q):
        """! Computes the position of end effector points in the robot base frame.

        @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
        @param q            The robot angular configuration (i.e. actuators values).

        @return  The positions of end effector points in the robot base frame.
        """

        #Robot base
        transformation_world_base = np.eye(4)
        transformation_world_base = transformation_world_base.at[:3,-1].set(parameters[:3])
        transformation_world_base = transformation_world_base.at[:3,:3].set(R.from_mrp(parameters[3:6]).as_matrix())

        #Robot geometric parameters
        pi = parameters[6:6+len(self.pi_N)]

        #Robot flange and tool
        transformation_flange_tool = np.empty(np.shape(self.transformation_flange_tool_N))
        for i,P in enumerate(np.reshape(parameters[6+len(self.pi_N):],(-1,3))):
            tmp = np.dot(self.transformation_flange_N,self.transformation_flange_tool_N[i])
            tmp = tmp.at[:3,-1].set(P)
            transformation_flange_tool = transformation_flange_tool.at[i].set(tmp)

        T = transformation_world_base
        for i in range(self.joints_number):
            T = np.dot(T,self.Ti(i,q,pi))
        P = np.vstack([np.dot(T,t)[:3,-1] for t in transformation_flange_tool])
        return(P)
    
    def P_full_func(self, parameters, q):
        """! Computes the global homogeneous transformation matrix between the robot base frame and the robot flange frame.

        @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
        @param q            The robot angular configuration (i.e. actuators values).

        @return  The global homogeneous transformation matrix between the robot base frame and the robot flange frame.
        """
        
        #Robot base
        transformation_world_base = np.eye(4)
        transformation_world_base = transformation_world_base.at[:3,-1].set(parameters[:3])
        transformation_world_base = transformation_world_base.at[:3,:3].set(R.from_mrp(parameters[3:6]).as_matrix())

        #Robot geometric parameters
        pi = parameters[6:6+len(self.pi_N)]

        #Robot flange and tool
        transformation_flange_tool = np.eye(4)
        transformation_flange_tool = transformation_flange_tool.at[:3,-1].set(parameters[6+len(self.pi_N):9+len(self.pi_N)])
        transformation_flange_tool = transformation_flange_tool.at[:3,:3].set(R.from_mrp(parameters[9+len(self.pi_N):]).as_matrix())

        T = transformation_world_base
        for i in range(self.joints_number):
            T = np.dot(T,self.Ti(i,q,pi))
        T = np.dot(T,transformation_flange_tool)

        P = np.hstack((T[:3,-1],R.from_matrix(T[:3,:3]).as_mrp()))
        return(P)

if __name__ == "__main__":

    @cloup.command()
    @load_geometric_model
    def test(**kwargs):
        robot_model = kwargs["robotModel"]
        print(robot_model.robot_name)
        print(robot_model.tool_name)
        print(robot_model.T(np.zeros(7)))

    test()