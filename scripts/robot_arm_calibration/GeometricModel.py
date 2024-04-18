#!/usr/bin/python3

#System packages
import os
import cloup
import functools
from abc import ABC, abstractmethod

#Utility packages
#from jax import jit    
#Using a just-in-time (JIT) compilation decorator, sequences of operations can be optimized together and run at once. Not all JAX code can be JIT compiled, as it requires array shapes to be static & known at compile time.
import jax.numpy as np
from jax.scipy.spatial.transform import Rotation as R

#Jacobian computation package
from jax import jacfwd, jacrev  
#jacfwd uses forward-mode automatic differentiation, which is more efficient for “tall” Jacobian matrices, while jacrev uses reverse-mode, which is more efficient for “wide” Jacobian matrices. For matrices that are near-square, jacfwd probably has an edge over jacrev.   

import sympy as sp
import dill

#Geometric parameters tools
from robot_arm_calibration.ParametersTools import importAllParametersFromNames, importAllParametersFromFile, getPackagePath

def load_geometric_model(func):
    @cloup.option("--robot_name", type=str)
    @cloup.option("--tool_name", type=str, default="")
    @cloup.option("--parameters_file", type=str)
    @cloup.option("--model_type", type=str, default="numeric")
    @cloup.constraint(cloup.constraints.RequireExactly(1),["robot_name", "parameters_file"])
    @functools.wraps(func)
    def wrapper(robot_name, tool_name, parameters_file, *args, **kwargs):
        parameters = {}
        if parameters_file:
            parameters = importAllParametersFromFile(parameters_file)
        else:
            parameters = importAllParametersFromNames(robot_name, tool_name)

        if kwargs["model_type"] == "numeric":
            robot_model = NumericRobotModel(**parameters)
        elif kwargs["model_type"] == "analytic":
            robot_model = AnalyticRobotModel(**parameters)
        else:
            raise ValueError("Invalid model type")
        
        return func(robot_model, *args, **kwargs)
    return wrapper

class RobotModel(ABC):

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
        self.transformation_flange_tool_N = kwargs["TToolN"]

        ### BASE OFFSET ###
        self.transformation_world_base = kwargs["TBaseOffset"]

        ### MODEL FUNCTIONS ###
        @abstractmethod
        def T_func(self, parameters, q):
            """! Computes the transformation matrix between the robot first frame and the robot flange frame (kinematic chain only).

            @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
            @param q            The robot angular configuration (i.e. actuators values).

            @return  The transformation matrix between the robot base frame and the robot flange frame.
            """
            pass

        @abstractmethod
        def P_func(self, parameters, q):    
            """! Computes the transformation matrix between the tool and the world frame (full) or the the position of the tool end effector points in the world frame (partial).

            @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
            @param q            The robot angular configuration (i.e. actuators values).

            @return  The transformation matrix between the robot base frame and the robot flange frame.
            """
            pass

        @abstractmethod
        def J_func(self, parameters, q):
            """! Computes the jacobian of P_func.

            @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
            @param q            The robot angular configuration (i.e. actuators values).

            @return  The transformation matrix between the robot base frame and the robot flange frame.
            """
            pass

class NumericRobotModel(RobotModel):
        
    def __init__(self, **kwargs):
        """! Initializes the robot numeric geometric model."""

        super().__init__(**kwargs)
    
        if(self.tool_points_number > 1):
            self.P_func = self.P_partial_func
        else:
            self.P_func = self.P_full_func

        self.J_func = jacfwd(self.P_func)

    def T_i(self, i, q, pi = None):
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

        T_i = np.eye(4)
        angles = np.zeros(3)

        ###Joint rotation
        angles = angles.at[np.abs(self.joints_axis[i])-1].set(np.sign(self.joints_axis[i])*q[i])
        T_i = T_i.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())
        T = np.dot(T,T_i)


        ###Joint offset
        if(self.joints_offset[i] == 1):
            T_i = np.eye(4)
            angles = np.zeros(3)

            angles = angles.at[np.abs(self.joints_axis[i])-1].set(parameters[0])
            T_i = T_i.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())
            T = np.dot(T,T_i)

        ###Link translation
        for n,axis in enumerate(self.links_translations[i]):
            T_i = np.eye(4)
            T_i = T_i.at[np.abs(axis)-1,-1].set(np.sign(axis)*parameters[self.joints_offset[i] + n])

            T = np.dot(T,T_i)

        ###Link rotation
        for n,axis in enumerate(self.links_rotations[i]):
            T_i = np.eye(4)
            angles = np.zeros(3)

            angles = angles.at[np.abs(axis)-1].set(np.sign(axis)*parameters[self.joints_offset[i] + len(self.links_translations[i]) + n])
            T_i = T_i.at[:3,:3].set(R.from_euler("xyz",angles).as_matrix())

            T = np.dot(T,T_i)

        return(T)

    def T(self, q, pi = None, transformation_world_base = None, transformation_flange_tool = None):
        """! Computes the transformation matrix between the world reference frame and the robot tool.

        @param q        The robot angular configuration (i.e. actuators values).
        @param pi       The robot kinematic parameters vector.
        @param transformation_world_base    The homogeneous transformation matrix between the world reference frame and the robot base.
        @param transformation_flange_tool    The homogeneous transformation matrix between the last robot joint and the robot tool.

        @return  The transformation matrix.
        """

        if pi is None:
            pi = self.pi_N
        if transformation_world_base is None:
            transformation_world_base = self.transformation_base_N
        if transformation_flange_tool is None:
            transformation_flange_tool = self.transformation_flange_N

        T = transformation_world_base
        for i in range(self.joints_number):
            T = np.dot(T,self.T_i(i,q,pi))
        return(np.dot(T,transformation_flange_tool))
    
    def P(self, q, pi = None, transformation_world_base = None, transformation_flange_tool = None):
        """! Computes the position of end effector points in the world reference frame.

        @param q        The robot angular configuration (i.e. actuators values).
        @param pi       The robot kinematic parameters vector.
        @param transformation_world_base    The homogeneous transformation matrix between the world reference frame and the robot base frame.
        @param transformation_flange_tool    The homogeneous transformation matrix between the last robot joint and the robot tool.

        @return  The positions of end effector points
        """

        P = np.vstack([self.T(q,pi,transformation_world_base,t)[:3,-1] for t in transformation_flange_tool])
        return(P)
    
    def T_func(self, parameters, q):
        """! Computes the transformation matrix between the robot first frame and the robot flange frame (kinematic chain only).

        @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
        @param q            The robot angular configuration (i.e. actuators values).

        @return  The transformation matrix between the robot base frame and the robot flange frame.
        """
        
        pi = parameters[6:6+len(self.pi_N)]
        return(self.T(q,pi,np.eye(4),np.eye(4)))
    
    def P_partial_func(self, parameters, q):
        """! Computes the position of the tool end effector points in the world frame.

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

        return(self.P(q,pi,transformation_world_base,transformation_flange_tool))
    
    def P_full_func(self, parameters, q):
        """! Computes the transformation matrix between the tool and the world frame.

        @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
        @param q            The robot angular configuration (i.e. actuators values).

        @return  The transformation matrix between the robot base frame and the robot flange frame.
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

        T = self.T(q,pi,transformation_world_base,transformation_flange_tool)
        P = np.hstack((T[:3,-1],R.from_matrix(T[:3,:3]).as_mrp()))
        return(P)

class AnalyticRobotModel(RobotModel):

    @staticmethod
    def rot_axis(angle,axis_index):
        """! Computes the analytic matrix representation of a rotation given an angle and an axis.

        @param angle        The analytic rotation angle.
        @param axisIndex    The index of the rotation axis : 1 -> X, 2 -> Y, 3 -> Z.

        @return  The analytic matrix representation of the rotation.
        """
        if(axis_index == 1):
            return(sp.rot_axis1(angle).T)
        elif(axis_index == 2):
            return(sp.rot_axis2(angle).T)
        elif(axis_index == 3):
            return(sp.rot_axis3(angle).T)
        else:
            return(sp.Identity(3))
    
    @staticmethod
    def matrix_to_rotvec(matrix):
        """! Converts a rotation matrix into a rotation vector representation.

        @param matrix   The rotation matrix.
        @return         The rotation vector representation.
        """
            
        angle = sp.acos((matrix.trace() - 1)/2)
        x = (matrix[2,1] - matrix[1,2])/sp.sqrt((matrix[2,1] - matrix[1,2])**2+(matrix[0,2] - matrix[2,0])**2+(matrix[1,0] - matrix[0,1])**2)
        y = (matrix[0,2] - matrix[2,0])/sp.sqrt((matrix[2,1] - matrix[1,2])**2+(matrix[0,2] - matrix[2,0])**2+(matrix[1,0] - matrix[0,1])**2)
        z = (matrix[1,0] - matrix[0,1])/sp.sqrt((matrix[2,1] - matrix[1,2])**2+(matrix[0,2] - matrix[2,0])**2+(matrix[1,0] - matrix[0,1])**2)

        return(angle*sp.Matrix([x,y,z]))

    @staticmethod
    def rotvec_to_matrix(rotvec):
        """! Converts a rotation vector representation into a rotation matrix
        
        @param rotvec   The rotation vector representation.
        @return         The rotation matrix.
        """
        
        (x, y, z) = rotvec
        angle = sp.sqrt(x**2 + y**2 + z**2)
        (x, y, z) = (x / angle, y / angle, z / angle)
        s = sp.sin(angle * sp.core.singleton.S.Half)
        a = sp.cos(angle * sp.core.singleton.S.Half)
        b = x * s
        c = y * s
        d = z * s

        return(sp.Quaternion(a, b, c, d).to_rotation_matrix())

    @staticmethod
    def quat_to_mrp(quat):
        """! Converts a quaternion into a MRP representation.

        @param quat     The quaternion.
        @return         The MRP representation.
        """
            
        quat_matrix = quat.to_Matrix()
        sign = -1 if quat_matrix[3] < 0 else 1
        denominator = 1 + sign * quat_matrix[3]
        return((sign/denominator) * sp.Matrix([quat_matrix[0],quat_matrix[1],quat_matrix[2]]))

    @staticmethod
    def mrp_to_matrix(mrp):
        """! Converts a MRP representation into a quaternion.

        @param mrp     The MRP representation.
        @return         The quaternion.
        """
            
        mrp_squared_plus_1 = mrp.dot(mrp) + 1
        a = 2*mrp[0]/mrp_squared_plus_1
        b = 2*mrp[1]/mrp_squared_plus_1
        c = 2*mrp[2]/mrp_squared_plus_1
        d = (2 - mrp_squared_plus_1)/mrp_squared_plus_1
        return(sp.Quaternion(a, b, c, d).to_rotation_matrix())
        
    def __init__(self, **kwargs):
        """! Initializes the robot numeric geometric model."""

        super().__init__(**kwargs)

        self.model_folder = getPackagePath() + "/config/nominalModels/robots/" + self.robot_name + "/"
        self.model_suffix = self.robot_name + "_" + self.tool_name if (self.tool_name != 0) else self.robot_name

        #If the analytic model does not exist, it is generated
        if(not os.path.exists(self.model_folder + "Jacobian_" + self.model_suffix + ".pickle") or not os.path.exists(self.model_folder + "GeometricModel_" + self.model_suffix + ".pickle") or not os.path.exists(self.model_folder + "RawGeometricModel_" + self.robot_name + ".pickle")):

            ### Robot base ###
            P_base = sp.MatrixSymbol("P_base",3,1).as_mutable()
            Phi_base = sp.MatrixSymbol("Phi_base",3,1).as_mutable()

            T_base = sp.Identity(4).as_mutable()
            T_base[:3,:3] = self.mrp_to_matrix(Phi_base)
            T_base[:3,-1] = P_base

            ### Robot kinematic chain ###
            q = sp.MatrixSymbol("q",self.joints_number,1).as_mutable()  #Joints angles

            T = sp.Identity(4)
            pi = []
            for i in range(self.joints_number):
                
                #Joint rotation
                T_i = sp.Identity(4).as_mutable()
                T_i[:3,:3] = self.rot_axis(np.sign(self.joints_axis[i])*q[i,0],np.abs(self.joints_axis[i]))
                T = T*T_i

                #Joint offset
                if(self.joints_offset[i] == 1):
                    dq_i = sp.symbols("dq_"+str(i))
                    T_i = sp.Identity(4).as_mutable()
                    T_i[:3,:3] = self.rot_axis(dq_i,np.abs(self.joints_axis[i]))
                    T = T*T_i
                    pi.append(dq_i)

                #Link translation
                T_axis_i = sp.symbols("T_x_"+str(i)+",T_y_"+str(i)+",T_z_"+str(i))
                for axis in self.links_translations[i]:
                    T_i = sp.Identity(4).as_mutable()
                    T_i[np.abs(axis)-1,-1] = [[np.sign(axis)*T_axis_i[np.abs(axis)-1]]]
                    T = T*T_i
                    pi.append(T_axis_i[np.abs(axis)-1])

                #Link rotation
                R_axis_i = sp.symbols("R_x_"+str(i)+",R_y_"+str(i)+",R_z_"+str(i))
                for axis in self.links_rotations[i]:
                    T_i = sp.Identity(4).as_mutable()
                    T_i[:3,:3] = self.rot_axis(np.sign(axis)*R_axis_i[np.abs(axis)-1],np.abs(axis))
                    T = T*T_i
                    pi.append(R_axis_i[np.abs(axis)-1])

            ### Robot tool ###

            #Multiple tool points => Partial pose measurement (tool points position measurement)
            if(self.tool_points_number > 1):
                T_tool = []
                for i,_ in enumerate(self.transformation_flange_tool_N):
                    T_tool_i = sp.ones(4,1).as_mutable()
                    T_tool_i[:3,0] = sp.MatrixSymbol("P_tool"+str(i),3,1).as_mutable()
                    T_tool.append(T_tool_i)
                T_tool = sp.Matrix.hstack(*T_tool)

            #Single tool point => Full pose measurement (tool position and orientation measurement)
            else:
                P_tool = sp.MatrixSymbol("P_Tool",3,1).as_mutable()
                Phi_tool = sp.MatrixSymbol("Phi_Tool",3,1).as_mutable()

                T_tool = sp.Identity(4).as_mutable()
                T_tool[:3,:3] = self.mrp_to_matrix(Phi_tool)
                T_tool[:3,-1] = P_tool

            ### Model parameters ###
            pi = sp.Matrix(pi)
            if(self.tool_points_number > 1):
                parameters = sp.Matrix.vstack(P_base,Phi_base,pi,T_tool[:3,:].T.reshape(3*self.tool_points_number,1))
            else:
                parameters = sp.Matrix.vstack(P_base,Phi_base,pi,P_tool,Phi_tool)

            print("Computing jacobian...")

            P = (T_base*T)*T_tool
            if(self.tool_points_number > 1):
                P = P[:3,:].T.reshape(3*self.tool_points_number,1)
            else:
                P = sp.Matrix.vstack(P[:3,-1],self.quat_to_mrp(sp.Quaternion.from_rotation_matrix(P[:3,:3])))
            J = P.jacobian(parameters)

            print("Computation done !")

            dill.settings['recurse'] = True

            print("Turning sympy expressions to lambda functions...")

            self.T_raw = sp.lambdify((parameters,q),T,modules=["numpy","scipy"])
            with open(self.model_folder + "RawGeometricModel_" + self.robot_name + ".pickle", "wb+") as file:
                dill.dump(self.T_raw, file)
            self.P_raw = sp.lambdify((parameters,q),P,modules=["numpy","scipy"])
            with open(self.model_folder + "GeometricModel_" + self.model_suffix + ".pickle", "wb+") as file:
                dill.dump(self.P_raw, file)
            self.J_raw = sp.lambdify((parameters,q),J,modules=["numpy","scipy"])
            with open(self.model_folder + "Jacobian_" + self.model_suffix + ".pickle", "wb+") as file:
                dill.dump(self.J_raw, file)

            print("Done !")   
            
        #If the analytic model exists, it is loaded
        else:
            print("Loading geometric model and jacobian...")

            self.T_raw = dill.load(open(self.model_folder + "RawGeometricModel_" + self.robot_name + ".pickle", "rb"))
            self.P_raw = dill.load(open(self.model_folder + "GeometricModel_" + self.model_suffix + ".pickle", "rb"))
            self.J_raw = dill.load(open(self.model_folder + "Jacobian_" + self.model_suffix + ".pickle", "rb"))

            print("Done !")

    def T_func(self,parameters,q):
        """! Computes the transformation matrix between the robot first frame and the robot flange frame (kinematic chain only).

        @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
        @param q            The robot angular configuration (i.e. actuators values).

        @return  The transformation matrix between the robot base frame and the robot flange frame.
        """
        
        T = self.T_raw(parameters,q)
        for i,row in enumerate(T):
            for j,_ in enumerate(row):
                try:
                    T[i,j] = T[i,j].item()
                except:
                    T[i,j] = float(T[i,j])
        return(T)

    def P_func(self,parameters,q):
        """! Computes the transformation matrix between the tool and the world frame (full) or the the position of the tool end effector points in the world frame (partial).

        @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
        @param q            The robot angular configuration (i.e. actuators values).

        @return  The transformation matrix between the robot base frame and the robot flange frame.
        """

        P = self.P_raw(parameters,q)
        for i,_ in enumerate(P):
            try:
                P[i] = P[i].item()
            except:
                P[i] = float(P[i])
        return(P)

    def J_func(self,parameters,q):
        """! Computes the jacobian of P_func.

        @param parameters   The robot identifiable kinematic parameters vector (base, link/joints, tools).
        @param q            The robot angular configuration (i.e. actuators values).

        @return  The transformation matrix between the robot base frame and the robot flange frame.
        """

        J = self.J_raw(parameters,q)
        for i,row in enumerate(J):
            for j,_ in enumerate(row):
                try:
                    J[i,j] = J[i,j].item()
                except:
                    J[i,j] = float(J[i,j])
        return(J)

if __name__ == "__main__":

    @cloup.command()
    @load_geometric_model
    def test(robot_model, **kwargs):
        print(robot_model.robot_name)
        print(robot_model.tool_name)
        print(robot_model.T(np.zeros(7)))

    test()