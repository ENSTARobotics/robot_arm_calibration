#!/usr/bin/python3

#System packages
from importlib import import_module
import sys
import os
from copy import deepcopy
import yaml

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import basinhopping
import re
import scipy.odr as odr

#Timing packages
import time

### Retrieve parameters from directory name
currentParameters = np.array(re.split('/|_',os.getcwd()))
currentParameters = currentParameters[np.where(currentParameters == "calibratedModels")[0][0] + 1:]
robotName = currentParameters[0]
markerName = currentParameters[1]

### Get robot config directory path
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/nominalModels/" + robotName + "/"
sys.path.append(configPath)

### Import robot parameters from dedicated file
try:
    module = import_module(robotName + "_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("Invalid robot name for robot parameters importation ! Are you in a proper measurements folder ? ")
    exit(1)

### Get marker config directory path
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/nominalModels/markers/"
sys.path.append(configPath)

### Import marker parameters from dedicated file
try:
    module = import_module(markerName + "_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("Invalid robot name for markers parameters importation ! Are you in a proper measurements folder ? ")
    exit(1)

#Custom packages (needing robot and marker parameters inclusions)
from CalibrationCSVParsing import calibrationCSVParsing
from AnalyticGeometricModel import Pfunc,Jfunc,size
from NumericGeometricModel import display

def computeDelta(parameters, Pij, Jointi):
    """! Computes the deltas between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints angles vector.

    @return  The deltas vector.
    """

    return(Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

def computeError(parameters, Pij, Jointi):
    """! Computes the error (i.e. euclidian distance) between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints angles vector.

    @return  The computed error.
    """

    delta = computeDelta(parameters,Pij,Jointi)
    return(np.dot(delta.T,delta))

def computeIdentificationJacobian(parameters,Jointi):
    """! Computes the identifiaction jacobian matrix.

    @param parameters   The current robot parameters vector.
    @param Jointi       The measured joints angles vector.

    @return  The identification jacbobian matrix.
    """

    return(np.vstack([Jfunc(parameters,q) for q in Jointi]))

def computeIdentificationGradient(parameters,Pij,Jointi):
    """! Computes the identifiaction gradient vector.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints angles vector.

    @return  The identification jacbobian matrix.
    """

    delta = computeDelta(parameters,Pij,Jointi)
    gradient = -2*np.dot(computeIdentificationJacobian(parameters,Jointi).T,delta)
    #gradient /= np.linalg.norm(gradient)
    return(gradient)

def computeLearningRate(parameters,Pij,Jointi):
    """! Computes the optimal learning rate for the gradient descend according to Wolfe conditions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints angles vector.

    @return  The optimal learning rate.
    """

    delta = computeDelta(parameters,Pij,Jointi)
    J = computeIdentificationJacobian(parameters,Jointi)
    JtJ = np.dot(J,J.T)
    alpha = np.dot(np.dot(delta.T,JtJ),delta)/(2*np.dot(np.dot(delta.T,np.dot(JtJ,JtJ)),delta))
    return(alpha)

globalAlgorithms = {"BH":"BFGS based Basin-Hopping"}

def globalIdentifyParameters(initialParameters,PijLear,JointiLear,maxIterations,algorithmName="BH",PijTest=None,JointiTest=None):
    """! Performs the robot model parameters identification using the specified global algorithm.

    @param initialParameters   The current robot parameters vector.
    @param PijLear      The learning measured positions vector.
    @param JointiLear   The learning measured joints angles vector.
    @param maxIterations  The maximum number of iterations- default = 10.
    @param PijTest      The testing measured positions vector.
    @param JointiTest   The testing measured joints angles vector.

    @return  The optimized parameters vector.
    """

    print(globalAlgorithms[algorithmName] + " algorithm")

    parameters = globals()[algorithmName.upper()](initialParameters,PijLear,JointiLear,maxIterations,PijTest,JointiTest)

    errorLear = computeError(parameters,PijLear,JointiLear)
    errorTest = None
    if(len(JointiTest)!=0):
        errorTest = computeError(parameters,PijTest,JointiLear)

    return(parameters,errorLear,errorTest)

def BH(initialParameters,PijLear,JointiLear,maxIterations,PijTest,JointiTest):
    """! Performs the robot model parameters identification using the BFGS based Basin-Hopping algorithm.

    @param initialParameters   The current robot parameters vector.
    @param PijLear      The learning measured positions vector.
    @param JointiLear   The learning measured joints angles vector.
    @param maxIterations  The maximum number of iterations.
    @param algorithm    The optimization algorithm to be used.
    @param PijTest      The testing measured positions vector.
    @param JointiTest   The testing measured joints angles vector.

    @return  The optimized parameters vector.
    """

    def function(parameters):
        return(computeError(parameters,PijLear,JointiLear),computeIdentificationGradient(parameters,PijLear,JointiLear))

    def callback(x,f,accept):
        #Intermediate error
        print("Intermediate error : " + str(f))

        if(len(JointiTest)!=0):
            print("Intermediate test error :  " + str(computeError(x,PijTest,JointiTest)))

        return(accept)

    result = basinhopping(function,x0=np.zeros(len(initialParameters)),niter=maxIterations,stepsize=10,minimizer_kwargs={"method":"L-BFGS-B","jac":True},disp=True,callback=callback)
    optimizedParameters = result.x

    return(optimizedParameters)

localAlgorithms = {"GN":"Gauss-Newton",
                   "GD":"Gradient descent",
                   "LM":"Levendberg-Marquardt",
                   "BFGS":"Broyden-Fletcher-Goldfarb-Shanno"}

def localIdentifyParameters(initialParameters,PijLear,JointiLear,stopDelta,algorithmName="GN",PijTest=None,JointiTest=None,maxIterations=10):
    """! Performs the robot model parameters identification using the specified locally iterative algorithm.

    @param initialParameters   The current robot parameters vector.
    @param PijLear      The learning measured positions vector.
    @param JointiLear   The learning measured joints angles vector.
    @param stopDelta    The relative delta stopping criterion.
    @param algorithm    The optimization algorithm to be used.
    @param PijTest      The testing measured positions vector.
    @param JointiTest   The testing measured joints angles vector.
    @param maxIterations  The maximum number of iterations - default = 10.

    @return  The identified parameters vector, final learning error and final testing error.
    """

    print(localAlgorithms[algorithmName] + " algorithm")

    #Errors initialisation
    errorLear = 0
    errorLearPrev = 0
    errorTest = None

    #Parameters initialisation
    parameters = initialParameters

    for i in range(maxIterations):

        print("Iteration " + str(i+1))

        parameters = globals()[algorithmName.upper()](parameters,PijLear,JointiLear)

        #Intermediate error
        errorLearPrev = errorLear
        errorLear = computeError(parameters,PijLear,JointiLear)
        print("Intermediate error :  " + str(errorLear))

        if(len(JointiTest)!=0):
            errorTest = computeError(parameters,PijTest,JointiLear)
            print("Intermediate test error :  " + str(errorTest))

        #Stopping criterion
        if(errorLearPrev > 1e-10 and errorLear/errorLearPrev < stopDelta):
            break

    return(parameters,errorLear,errorTest)

def GN(parameters,Pij,Jointi):
    """! Performs the robot model parameters optimization using the Gauss-Newton algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints angles vector.

    @return  The optimized parameters vector.
    """

    optimizedParameters = parameters + np.dot(np.linalg.pinv(computeIdentificationJacobian(parameters,Jointi)),computeDelta(parameters,Pij,Jointi))

    return(optimizedParameters)

def GD(parameters,Pij,Jointi):
    """! Performs the robot model parameters optimization using the gradient descent algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints angles vector.

    @return  The optimized parameters vector.
    """

    alpha = computeLearningRate(parameters,Pij,Jointi)
    optimizedParameters = parameters - alpha*computeIdentificationGradient(parameters,Pij,Jointi)

    return(optimizedParameters)

def LM(parameters,Pij,Jointi):
    """! Performs the robot model parameters optimization using the Levendberg-Marquardt algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints angles vector.

    @return  The optimized parameters vector.
    """

    delta = computeDelta(parameters,Pij,Jointi)
    J = computeIdentificationJacobian(parameters,Jointi)
    JtJ = np.dot(J,J.T)
    alpha = np.dot(np.dot(delta.T,JtJ),delta)/(2*np.dot(np.dot(delta.T,np.dot(JtJ,JtJ)),delta))

    tJJ = np.dot(J.T,J)
    optimizedParameters = parameters + np.dot(np.linalg.inv(tJJ + alpha*np.diag(np.diag(tJJ))),np.dot(J.T,delta))

    return(optimizedParameters)

def BFGS(parameters,Pij,Jointi):
    """! Performs the robot model parameters optimization using the Broyden-Fletcher-Goldfarb-Shanno algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints angles vector.

    @return  The optimized parameters vector.
    """

    delta = computeDelta(parameters,Pij,Jointi)
    J = computeIdentificationJacobian(parameters,Jointi)
    gradient = -2*np.dot(J.T,delta)

    try:
        #Compute direction
        global B
        d = -np.dot(np.linalg.inv(B),gradient)
    except:
        #Initialisation step
        B = np.eye(len(parameters))

        #Compute direction
        d = -np.dot(np.linalg.inv(B),gradient)

    #Compute learning rate
    alpha = np.dot(np.dot(delta.T,J),d)/np.dot(np.dot(np.dot(d.T,J.T),J),d)

    #Update parameters
    deltaParameters = alpha*d
    optimizedParameters = parameters + alpha*d

    #Update B matrix
    delta = computeDelta(optimizedParameters,Pij,Jointi)
    J = computeIdentificationJacobian(optimizedParameters,Jointi)
    newGradient = -2*np.dot(J.T,delta)

    deltaGradient = np.array([newGradient - gradient]).T
    B += np.dot(deltaGradient,deltaGradient.T)/np.dot(deltaGradient.T,deltaParameters) - np.dot(np.dot(np.dot(B,deltaParameters),deltaParameters.T),B)/np.dot(np.dot(deltaParameters.T,B),deltaParameters)

    return(optimizedParameters)

### Main loop
if __name__ == "__main__":

    #Timer initialisation
    t0 = time.time()

    ### Retrieve measurements data
    itemsNames = EEPointsItemsNames + ["jointsAngles"]

    Items = calibrationCSVParsing(itemsNames,"Calibration.csv")

    ### Robot base, robot parameters and end effector points transformations estimations
    TBase0 = TBaseN

    TTool0 = np.empty((N_EEPOINTS,4,4))
    for j in range(N_EEPOINTS):
        TTool0[j] = np.dot(TFlangeN,TToolN_marker[j])

    PBase0 = TBase0[:3,-1]
    RBase0 = R.from_matrix(TBase0[:3,:3])
    PTool0 = TTool0[:,:3,-1]
    pi0 = pi0

    #Merging all parameters together
    parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten())) 

    ### Create learning and testing subsets
    N_MEASUREMENTS = len(Items["jointsAngles"])
    N_LEARNING = min(N_MEASUREMENTS,2*len(parameters0))
    N_TESTS = 0

    print("Number of experiments : " + str(N_LEARNING))
    print("Number of tests : " + str(N_TESTS))

    print("Test on first measurement :")
    for i,item in enumerate(EEPointsItemsNames):
        print("End effector point " + item)
        print(Items[item][0])
        print(Pfunc(parameters0,Items["jointsAngles"][0])[3*i:3*i+3].T[0])

    PijLear = np.empty((N_LEARNING,N_EEPOINTS,3))
    JointiLear = np.empty((N_LEARNING,JOINTS_NUMBER))
    PijTest = np.empty((N_TESTS,N_EEPOINTS,3))
    JointiTest = np.empty((N_TESTS,JOINTS_NUMBER))

    randomIndices = np.random.permutation(N_LEARNING+N_TESTS)
    randomIndices = np.arange(N_LEARNING+N_TESTS)

    #Randomly shuffle measurements between the learning and testing subsets
    indexLear = randomIndices[:N_LEARNING]
    for i,item in enumerate(EEPointsItemsNames):
        PijLear[:,i,:] = Items[item][indexLear]
    JointiLear = Items["jointsAngles"][indexLear]

    if(N_TESTS != 0):
        indexTest = randomIndices[-N_TESTS:]
        for i,item in enumerate(EEPointsItemsNames):
            PijTest[:,i,:] = Items[item][indexTest]
        JointiTest = Items["jointsAngles"][indexTest]

    #Initial error
    initialError = computeError(parameters0,PijLear,JointiLear)
    print("Initial error : " + str(initialError))

    if(len(JointiTest)!=0):
        errorTest = computeError(parameters0,PijTest,JointiTest)
        print("Initial test error : " + str(errorTest))

    ### Retreive identification parameters
    algorithmName = "GN"
    algorithmType = "local"
    try : 
        algorithmType = (localAlgorithms.keys()+globalAlgorithms.keys())[sys.argv[1]]
    except:
        print("No or invalid optimization method given, defaulting to " + algorithmType + " " + algorithmName)

    if(algorithmType == "local"):
        stopDelta = 1e-8
        try : 
            stopDelta = float(sys.argv[2])
        except:
            print("No stop delta criterion given, defaulting to " + str(stopDelta))

        parameters,finalError,finalTestError = localIdentifyParameters(parameters0,PijLear,JointiLear,stopDelta,algorithmName,PijTest,JointiTest)

    elif(algorithmType == "global"):
        maxIterations = 10
        try : 
            maxIterations = float(sys.argv[2])
        except:
            print("No maximum iterations number given, defaulting to " + str(maxIterations))

        parameters,finalError,finalTestError = globalIdentifyParameters(parameters0,PijLear,JointiLear,maxIterations,algorithmName,PijTest,JointiTest)

    #Final error
    print("Final error : " + str(finalError))

    if(len(JointiTest)!=0):
        print("Final test error : " + str(finalTestError))

    ### FILL ROBOT CALIBRATED MODEL

    print("Saving robot calibrated model...")

    calibratedModel = {}

    #Joints axis
    calibratedModel["jointsAxis"] = JOINTS_AXIS.tolist()

    #Robot base
    PBase = np.array([parameters[:3]]).T
    RBase = R.from_euler("xyz",parameters[3:6]).as_matrix()
    TBase = np.vstack((np.hstack((RBase,PBase)),np.array([[0,0,0,1]])))

    calibratedModel["robotBase"] = TBase.flatten().tolist()

    #Robot geometric parameters
    pi = parameters[6:6+len(pi0)]

    piFull = np.zeros(JOINTS_NUMBER*7)

    PARAMETERS_INDEX = np.array([sum(JOINTS_OFFSET[:i]) + size(LINKS_TRANSLATIONS[:i]) + size(LINKS_ROTATIONS[:i]) for i in range(JOINTS_NUMBER+1)])

    for i,jointAxis in enumerate(JOINTS_AXIS):
        localParameters = pi[PARAMETERS_INDEX[i]:PARAMETERS_INDEX[i+1]]
        currentIndex = 0

        #Joint offset
        if(JOINTS_OFFSET[i] == 1):
            piFull[i*7] = localParameters[0]
            currentIndex += 1

        #Link translations
        for k,axis in enumerate(LINKS_TRANSLATIONS[i]):
            piFull[i*7+1:i*7+4][np.abs(axis)-1] = localParameters[currentIndex]
            currentIndex += 1

        #Link rotations
        for k,axis in enumerate(LINKS_ROTATIONS[i]):
            piFull[i*7+4:i*7+7][np.abs(axis)-1] = localParameters[currentIndex]
            currentIndex += 1

    calibratedModel["robotParameters"] = piFull.tolist()

    #Robot flange

    TTool = np.empty(np.shape(TToolN_marker))
    for i,P in enumerate(np.reshape(parameters[6+len(pi0):],(-1,3))):
        TTool[i] = np.dot(TFlangeN,TToolN_marker[i])
        TTool[i][:3,-1] = P
        TTool[i] = np.dot(TTool[i],np.linalg.inv(TToolN_marker[i]))
    TTool = np.average(TTool,axis=0)

    calibratedModel["robotFlange"] = TTool.flatten().tolist()

    ### FILL SETUP CONFIGURATION

    """
    calibratedModel["baseReference"] = np.eye(4)

    calibratedModel["toolReferences"] = {}
    for i,tool in enumerate(TToolN_marker):
        calibratedModel["toolReferences"][str(i+1)] = np.dot(np.linalg.inv(TToolN_body),tool).flatten().tolist()

    calibratedModel["toolReferencesNumber"] = len(TToolN_marker)
    """

    calibratedModel["initialError"] = float(initialError)
    calibratedModel["finalError"] = float(finalError)

    if(not os.path.isdir(os.getcwd() + "/" + algorithmName.upper() + "_" + str(Niterations) + "/")):
        os.mkdir(os.getcwd() + "/" + algorithmName.upper() + "_" + str(Niterations) + "/")
    with open(os.getcwd() + "/" + algorithmName.upper() + "_" + str(Niterations) + "/CalibratedModel.yaml",'w') as file:
        yaml.dump(calibratedModel, file)
    print("Done !")

    print("-- Parameters identified in " + str(time.time() - t0) + " seconds --")

