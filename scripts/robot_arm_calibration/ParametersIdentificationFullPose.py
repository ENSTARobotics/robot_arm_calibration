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
from robot_arm_calibration.ParametersTools import getIdentifiableParametersPartial, saveParameters

#Custom packages (needing robot and marker parameters inclusions)
from robot_arm_calibration.CalibrationCSVParsing import calibrationCSVParsing
from robot_arm_calibration.NumericGeometricModel import Tfunc, Pfunc, Jfunc, TBaseN, TFlangeN, TToolN, piN, EEItemsNames, N_EEPOINTS, JOINTS_NUMBER

def simplePreconditionning(fullPose):
    #return(np.hstack([fullPose[:3],R.from_quat(fullPose[3:]).as_mrp()]))
    L = 1
    return(np.hstack([fullPose[:3],L*fullPose[3:]]))

def computeDelta(parameters, Pij, Jointi, preconditioning = simplePreconditionning):
    """! Computes the deltas between the measured and modeled data.

    @param parameters       The current robot parameters vector.
    @param Pij              The measured data vector.
    @param Jointi           The measured joints data vector.
    @param preconditioning   The preconditioning function to be used (coordinates transformation and weighting) - default : simplePreconditionning.

    @return  The deltas vector.
    """

    computedData = np.vstack([Pfunc(parameters,q) for q in Jointi])

    Pij = preconditioning(Pij)
    computedData = preconditioning(computedData)

    return(Pij.flatten() - computedData.flatten())

def computeError(parameters, Pij, Jointi):
    """! Computes the error (i.e. euclidian distance) between the measured and modeled data.

    @param parameters   The current robot parameters vector.
    @param Pij          The measured data vector.
    @param Jointi       The measured joints data vector.

    @return  The computed error.
    """

    delta = computeDelta(parameters,Pij,Jointi)
    return(np.dot(delta.T,delta))

globalAlgorithms = {"BH":"BFGS based Basin-Hopping"}

def globalIdentifyParameters(initialParameters,PijLear,JointiLear,maxIterations=10,algorithmName="BH",PijTest=None,JointiTest=None):
    """! Performs the robot model parameters identification using the specified global algorithm.

    @param initialParameters    The current robot parameters vector.
    @param PijLear              The learning measured data vector.
    @param JointiLear           The learning measured joints data vector.
    @param maxIterations        The maximum number of iterations- default = 10.
    @param PijTest              The testing measured data vector.
    @param JointiTest           The testing measured joints data vector.

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
    @param PijLear              The learning measured data vector.
    @param JointiLear           The learning measured joints data vector.
    @param maxIterations        The maximum number of iterations.
    @param algorithm            The optimization algorithm to be used.
    @param PijTest              The testing measured data vector.
    @param JointiTest           The testing measured joints data vector.

    @return  The optimized parameters vector.
    """

    def function(parameters):
        return(computeError(parameters,PijLear,JointiLear))

    def callback(x,f,accept):
        #Intermediate error
        print("Intermediate error : " + str(f))

        if(len(JointiTest)!=0):
            print("Intermediate test error :  " + str(computeError(x,PijTest,JointiTest)))

        return(accept)

    result = basinhopping(function,x0=np.zeros(len(initialParameters)),niter=maxIterations,stepsize=10,minimizer_kwargs={"method":"L-BFGS-B","jac":False},disp=True,callback=callback)
    optimizedParameters = result.x

    return(optimizedParameters)

#MERGE feature/new-algorithms !

### Main loop
if __name__ == "__main__":

    #Timer initialisation
    #import time
    #t0 = time.time()

    ### Retrieve measurements data
    print("Retrieving calibration measurement data...")
    itemsNames = EEItemsNames + ["jointsdata"]

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
    parameters0 = getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN)

    #Estimate the transformation between the sensor reference and the robot base
    if(estimateTBaseN != 0):

        EEPoints0 = []
        EEPoints1 = []
        for i,configuration in enumerate(Items["jointsdata"]):
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
            for i,jointsPosition in enumerate(Items["jointsdata"]):
                for j,item in enumerate(EEItemsNames):
                    Items[item][i] = np.dot(TBaseSensor,np.append(Items[item][i],1).T)[:3]
                    np.savetxt(file,np.round(Items[item][i].reshape(1,-1),6),delimiter=",",fmt='%1.6f')
                np.savetxt(file,np.round(jointsPosition.reshape(1,-1),6),delimiter=",",fmt='%1.6f')
                file.write(str.encode("---\n"))

        #print(np.dot(TSensorBase,np.hstack([EEPoints0,np.ones((len(EEPoints0),1))]).T).T[:,:-1] - EEPoints1)

    ### Create learning and testing subsets
    N_MEASUREMENTS = len(Items["jointsdata"])
    N_LEARNING = min(N_MEASUREMENTS,2*len(parameters0))
    N_TESTS = 0

    print("Number of experiments : " + str(N_LEARNING))
    print("Number of tests : " + str(N_TESTS))

    print("Test on first measurement :")
    for i,item in enumerate(EEItemsNames):
        print("Delta end effector point " + item)
        print(Items[item][0] - Pfunc(parameters0,Items["jointsdata"][0])[3*i:3*i+3].T[0])

    print("---")

    randomIndex = np.random.randint(0,N_MEASUREMENTS)
    print("Test on random measurement (" + str(randomIndex) + ") :")
    randomIndex = np.random.randint(0,N_MEASUREMENTS)
    for i,item in enumerate(EEItemsNames):
        print("End effector point " + item)
        print(Items[item][randomIndex] - Pfunc(parameters0,Items["jointsdata"][randomIndex])[3*i:3*i+3].T[0])

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
    JointiLear = Items["jointsdata"][indexLear]

    if(N_TESTS != 0):
        indexTest = randomIndices[-N_TESTS:]
        for i,item in enumerate(EEItemsNames):
            PijTest[:,i,:] = Items[item][indexTest]
        JointiTest = Items["jointsdata"][indexTest]

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
    RBase = R.from_mrp(parameters[3:6]).as_matrix()
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

