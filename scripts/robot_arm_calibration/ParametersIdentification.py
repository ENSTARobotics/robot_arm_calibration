#!/usr/bin/python3

#System packages
import sys
import os
import yaml

#Utility packages
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import basinhopping

### Import robot and marker parameters
from robot_arm_calibration.ParametersTools import getIdentificationParameters, saveParameters

#Custom packages (needing robot and marker parameters inclusions)
from robot_arm_calibration.CalibrationCSVParsing import calibrationCSVParsing
from robot_arm_calibration.AnalyticGeometricModel import Tfunc, Pfunc, Jfunc, TBaseN, TFlangeN, TToolN, piN, EEItemsNames, N_EEPOINTS, JOINTS_NUMBER

def computeDelta(parameters, Pij, Jointi):
    """! Computes the deltas between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The deltas vector.
    """

    return(Pij.flatten() - np.hstack([Pfunc(parameters,q).flatten() for q in Jointi]))

def computeError(parameters, Pij, Jointi):
    """! Computes the error (i.e. euclidian distance) between the measured and modeled positions.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The computed error.
    """

    delta = computeDelta(parameters,Pij,Jointi)
    return(np.dot(delta.T,delta))

def computeIdentificationJacobian(parameters,Jointi):
    """! Computes the identifiaction jacobian matrix.

    @param parameters   The current robot parameters vector.
    @param Jointi       The measured joints positions vector.

    @return  The identification jacbobian matrix.
    """

    return(np.vstack([Jfunc(parameters,q) for q in Jointi]))

def computeIdentificationGradient(parameters,Pij,Jointi):
    """! Computes the identifiaction gradient vector.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

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
    @param Jointi       The measured joints positions vector.

    @return  The optimal learning rate.
    """

    delta = computeDelta(parameters,Pij,Jointi)
    J = computeIdentificationJacobian(parameters,Jointi)
    JtJ = np.dot(J,J.T)
    alpha = np.dot(np.dot(delta.T,JtJ),delta)/(2*np.dot(np.dot(delta.T,np.dot(JtJ,JtJ)),delta))
    return(alpha)

globalAlgorithms = {"BH":"BFGS based Basin-Hopping"}

def globalIdentifyParameters(initialParameters,PijLear,JointiLear,maxIterations=10,algorithmName="BH",PijTest=None,JointiTest=None):
    """! Performs the robot model parameters identification using the specified global algorithm.

    @param initialParameters    The current robot parameters vector.
    @param PijLear              The learning measured positions vector.
    @param JointiLear           The learning measured joints positions vector.
    @param maxIterations        The maximum number of iterations- default = 10.
    @param PijTest              The testing measured positions vector.
    @param JointiTest           The testing measured joints positions vector.

    @return  The optimized parameters vector.
    """

    print(globalAlgorithms[algorithmName] + " algorithm")

    parameters = globals()[algorithmName](initialParameters,PijLear,JointiLear,maxIterations,PijTest,JointiTest)

    errorLear = computeError(parameters,PijLear,JointiLear)
    errorTest = None
    if(len(JointiTest)!=0):
        errorTest = computeError(parameters,PijTest,JointiLear)

    return(parameters,errorLear,errorTest)

def BH(initialParameters,PijLear,JointiLear,maxIterations,PijTest,JointiTest):
    """! Performs the robot model parameters identification using the BFGS based Basin-Hopping algorithm.

    @param initialParameters    The current robot parameters vector.
    @param PijLear              The learning measured positions vector.
    @param JointiLear           The learning measured joints positions vector.
    @param maxIterations        The maximum number of iterations.
    @param algorithm            The optimization algorithm to be used.
    @param PijTest              The testing measured positions vector.
    @param JointiTest           The testing measured joints positions vector.

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

def localIdentifyParameters(initialParameters,PijLear,JointiLear,stopDelta=1e-8,algorithmName="GN",PijTest=None,JointiTest=None,maxIterations=10):
    """! Performs the robot model parameters identification using the specified locally iterative algorithm.

    @param initialParameters   The current robot parameters vector.
    @param PijLear      The learning measured positions vector.
    @param JointiLear   The learning measured joints positions vector.
    @param stopDelta    The relative delta stopping criterion.
    @param algorithm    The optimization algorithm to be used.
    @param PijTest      The testing measured positions vector.
    @param JointiTest   The testing measured joints positions vector.
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

        parameters = globals()[algorithmName](parameters,PijLear,JointiLear)

        #Intermediate error
        errorLearPrev = errorLear
        errorLear = computeError(parameters,PijLear,JointiLear)
        print("Intermediate error :  " + str(errorLear))

        if(len(JointiTest)!=0):
            errorTest = computeError(parameters,PijTest,JointiLear)
            print("Intermediate test error :  " + str(errorTest))

        #Stopping criterion
        #print(1 - errorLear/errorLearPrev)
        if(errorLearPrev > 1e-10 and 1 - errorLear/errorLearPrev < stopDelta):
            break

    return(parameters,errorLear,errorTest)

def GN(parameters,Pij,Jointi):
    """! Performs the robot model parameters optimization using the Gauss-Newton algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The optimized parameters vector.
    """

    optimizedParameters = parameters + np.dot(np.linalg.pinv(computeIdentificationJacobian(parameters,Jointi)),computeDelta(parameters,Pij,Jointi))

    return(optimizedParameters)

def GD(parameters,Pij,Jointi):
    """! Performs the robot model parameters optimization using the gradient descent algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

    @return  The optimized parameters vector.
    """

    alpha = computeLearningRate(parameters,Pij,Jointi)
    optimizedParameters = parameters - alpha*computeIdentificationGradient(parameters,Pij,Jointi)

    return(optimizedParameters)

def LM(parameters,Pij,Jointi):
    """! Performs the robot model parameters optimization using the Levendberg-Marquardt algorithm.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured positions vector.
    @param Jointi       The measured joints positions vector.

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
    @param Jointi       The measured joints positions vector.

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
    #import time
    #t0 = time.time()

    ### Retrieve measurements data
    print("Retrieving calibration measurement data...")
    itemsNames = EEItemsNames + ["jointsPositions"]

    try:
        Items = calibrationCSVParsing(itemsNames,"Calibration.csv")
    except:
        print("No calibration measurements CSV ! Are you in a proper calibrated model folder ? ")
        sys.exit(-1)
    print("Done !")

    estimateTBaseN = 0
    try:
        estimateTBaseN = int(sys.argv[3])
    except:
        print("Skipping robot base transform estimation...")

    ### Robot base, robot parameters and end effector points transformations estimations
    parameters0 = getIdentificationParameters(TBaseN,TFlangeN,TToolN,piN)

    #Estimate the transformation between the sensor reference and the robot base
    if(estimateTBaseN != 0):

        EEPoints0 = []
        EEPoints1 = []
        for i,configuration in enumerate(Items["jointsPositions"]):
            TBaseFlange = np.dot(TBaseN,np.dot(Tfunc(parameters0,configuration),TFlangeN))
            EEPoints0.extend(np.vstack([np.dot(np.dot(TBaseFlange,TTool),np.array([[0,0,0,1]]).T)[:-1].T for TTool in TToolN]))
            for j,item in enumerate(EEItemsNames):
                EEPoints1.append(Items[item][i])

        EEPoints0 = np.array(EEPoints0)
        EEPoints1 = np.array(EEPoints1)
    
        #Does not enforce rotation matrix constraints
        #TSensorBase = np.dot(np.hstack([EEPoints1,np.ones((len(EEPoints1),1))]).T,np.linalg.pinv(np.hstack([np.dot(np.dot(TFlangeBase,TTool),np.array([[0,0,0,1]]).T) for TTool in TToolN])))

        #Enforces rotation matrix constraints !
        EEPoints1Centroid = np.mean(EEPoints1,axis=0)
        EEPoints0Centroid = np.mean(EEPoints0,axis=0)

        U,S,V = np.linalg.svd(np.dot((EEPoints0 - EEPoints0Centroid).T,(EEPoints1 - EEPoints1Centroid)))

        TSensorBase = np.eye(4)
        TSensorBase[:3,:3] = np.dot(np.dot(V.T,np.diag(np.append(np.ones(len(S)-1),np.linalg.det(np.dot(V.T,U.T))))),U.T)
        TSensorBase[:3,-1] = EEPoints1Centroid - np.dot(TSensorBase[:3,:3],EEPoints0Centroid)
        TBaseSensor = np.linalg.inv(TSensorBase)

        print("Sensor to base transformation estimation : \n" + str(TBaseSensor))

        #Transfering measurements in the estimated robot base frame !
        with open("CalibrationCorrected.csv",'wb+') as file:
            for i,jointsPosition in enumerate(Items["jointsPositions"]):
                for j,item in enumerate(EEItemsNames):
                    Items[item][i] = np.dot(TBaseSensor,np.append(Items[item][i],1).T)[:3]
                    np.savetxt(file,np.round(Items[item][i].reshape(1,-1),6),delimiter=",",fmt='%1.6f')
                np.savetxt(file,np.round(jointsPosition.reshape(1,-1),6),delimiter=",",fmt='%1.6f')
                file.write(str.encode("---\n"))

        #print(np.dot(TSensorBase,np.hstack([EEPoints0,np.ones((len(EEPoints0),1))]).T).T[:,:-1] - EEPoints1)

    ### Create learning and testing subsets
    N_MEASUREMENTS = len(Items["jointsPositions"])
    N_LEARNING = min(N_MEASUREMENTS,2*len(parameters0))
    N_TESTS = 0

    print("Number of experiments : " + str(N_LEARNING))
    print("Number of tests : " + str(N_TESTS))

    print("Test on first measurement :")
    for i,item in enumerate(EEItemsNames):
        print("Delta end effector point " + item)
        print(Items[item][0] - Pfunc(parameters0,Items["jointsPositions"][0])[3*i:3*i+3].T[0])

    print("---")

    randomIndex = np.random.randint(0,N_MEASUREMENTS)
    print("Test on random measurement (" + str(randomIndex) + ") :")
    randomIndex = np.random.randint(0,N_MEASUREMENTS)
    for i,item in enumerate(EEItemsNames):
        print("End effector point " + item)
        print(Items[item][randomIndex] - Pfunc(parameters0,Items["jointsPositions"][randomIndex])[3*i:3*i+3].T[0])

    tmp = input()

    PijLear = np.empty((N_LEARNING,N_EEPOINTS,3))
    JointiLear = np.empty((N_LEARNING,JOINTS_NUMBER))
    PijTest = np.empty((N_TESTS,N_EEPOINTS,3))
    JointiTest = np.empty((N_TESTS,JOINTS_NUMBER))

    randomIndices = np.random.permutation(N_LEARNING+N_TESTS)
    randomIndices = np.arange(N_LEARNING+N_TESTS)

    #Randomly shuffle measurements between the learning and testing subsets
    indexLear = randomIndices[:N_LEARNING]
    for i,item in enumerate(EEItemsNames):
        PijLear[:,i,:] = Items[item][indexLear]
    JointiLear = Items["jointsPositions"][indexLear]

    if(N_TESTS != 0):
        indexTest = randomIndices[-N_TESTS:]
        for i,item in enumerate(EEItemsNames):
            PijTest[:,i,:] = Items[item][indexTest]
        JointiTest = Items["jointsPositions"][indexTest]

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
        tmpAlgorithmName = sys.argv[1].upper()
        if(tmpAlgorithmName in localAlgorithms.keys()):
            algorithmType = "local"
        elif(tmpAlgorithmName in globalAlgorithms.keys()):
            algorithmType = "global"
        else:
            raise ValueError
        algorithmName = tmpAlgorithmName
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

    #Robot base
    PBase = np.array([parameters[:3]]).T
    RBase = R.from_rotvec(parameters[3:6]).as_matrix()
    TBase = np.vstack((np.hstack((RBase,PBase)),np.array([[0,0,0,1]])))

    #Robot geometric parameters
    pi = parameters[6:6+len(piN)]

    #Robot flange and tool
    TFlangeList = np.empty(np.shape(TToolN))
    TTool = np.empty(np.shape(TToolN))
    for i,P in enumerate(np.reshape(parameters[6+len(piN):],(-1,3))):
        TTool[i] = np.dot(TFlangeN,TToolN[i])
        TTool[i][:3,-1] = P
        TFlangeList[i] = np.dot(TTool[i],np.linalg.inv(TToolN[i]))
    TFlange = np.average(TFlangeList,axis=0)

    saveParameters(os.getcwd() + "/" + algorithmName + "/CalibratedModel.yaml", TBase = TBase, TFlange = TFlange, pi = pi, TTool = TToolN)

    #Additionnal parameters/information
    calibratedModel = {}
    calibratedModel["TLastLinkTools"] = TTool.flatten().tolist()
    if(estimateTBaseN != 0):
        calibratedModel["TBaseSensor"] = TBaseSensor.flatten().tolist()

    calibratedModel["initialError"] = float(initialError)
    calibratedModel["finalError"] = float(finalError)

    with open(os.getcwd() + "/" + algorithmName + "/CalibratedModel.yaml",'a') as file:
        yaml.dump(calibratedModel, file)
        
    print("Done !")

    #print("-- Parameters identified in " + str(time.time() - t0) + " seconds --")

