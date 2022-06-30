from importlib import import_module
from sys import argv, float_info
import os
import numpy as np
from numpy import diff

from AnalyticGeometricModel import Jfunc
from copy import deepcopy
import yaml

from pygad import GA,__version__

from scipy.spatial.transform import Rotation as R

from scipy.optimize import differential_evolution,Bounds

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(argv[1]+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID ROBOT NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

### IMPORT MARKER PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(argv[2]+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID MARKER NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

### RETREIVE NOMINAL PARAMETERS
TBase0 = TBaseN

TTool0 = np.empty((len(TTool0_marker),4,4))
for j in range(len(TTool0_marker)):
    TTool0[j] = np.dot(TToolN,TTool0_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

### SAMPLE ALL JOINTS RANGES
N = 10
try:
    N = int(argv[3])
except:
    print("No sampling value given, defaulting to " + str(N))
  
if(not os.path.exists("SampledConfigurations_" + argv[1] + "_" + str(N) + ".npy")):
    print("Computing sampled calibration configurations...")
    Q = np.empty(((N,JOINTS_NUMBER)))
    SampledCalibrationConfigurations = np.empty((N**JOINTS_NUMBER,JOINTS_NUMBER))
    for i in range(N):
        for j in range(JOINTS_NUMBER):
            Q[i,j] = jointsLowerLimits[j] + (jointsUpperLimits[j] - jointsLowerLimits[j])*i/(N-1)

    for i in range(N**JOINTS_NUMBER):
        SampledCalibrationConfigurations[i] = deepcopy(Q[0])
        for j in range(JOINTS_NUMBER):
            if((i+1)%N**(j) == 0):
                Q[:,j] = np.roll(Q[:,j],-1)

    print("Saving sampled calibration configurations...")
    with open("SampledConfigurations_" + argv[1] + "_" + str(N) + ".npy","wb") as f:
        np.save(f,SampledCalibrationConfigurations)
    print("Done !")

SampledCalibrationConfigurations = np.empty((0,0))
print("Loading sampled calibration configurations...")
with open("SampledConfigurations_" + argv[1] + "_" + str(N) + ".npy","rb") as f:
        SampledCalibrationConfigurations = np.load(f)
print("Done !")

targetPoseNumber = 2*len(parameters0)

flag = input("Random selection ? y/n")

if(flag == "y" or flag == "Y"):

    #RANDOM SELECTION

    randomIndices = np.random.permutation(N**JOINTS_NUMBER)
    CalibrationIndices = randomIndices[:targetPoseNumber]
    CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]

    print("Done !")

    print("Saving optimized calibration configurations...")
    with open("CalibrationConfigurations_" + argv[1] + "_" + argv[2] + "_random.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.flatten().tolist()}, file)
    print("Done !")

flag = input("Naive selection ? y/n")

if(flag == "y" or flag == "Y"):

    #NAIVE SELECTION

    naiveN = int(np.round(targetPoseNumber/JOINTS_NUMBER))
    CalibrationConfigurations = np.vstack([(jointsUpperLimits + jointsLowerLimits)/2]*(naiveN*JOINTS_NUMBER))

    for i in range(JOINTS_NUMBER):
        for j in range(naiveN):
            CalibrationConfigurations[i*naiveN + j][i] =  jointsLowerLimits[i] + (jointsUpperLimits[i] - jointsLowerLimits[i])*j/(naiveN-1)

    print("Done !")

    print("Saving optimized calibration configurations...")
    with open("CalibrationConfigurations_" + argv[1] + "_" + argv[2] + "_naive.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.flatten().tolist()}, file)
    print("Done !")

flag = input("Greedy algorithm ? y/n")

if(flag == "y" or flag == "Y"):

    ### GREEDY ALGORITHM

    print("Greedy algorithm...")

    randomIndices = np.random.permutation(N**JOINTS_NUMBER)
    CalibrationIndices = randomIndices[:targetPoseNumber]
    CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]

    NotRemovedIndices = deepcopy(CalibrationIndices)
    RemainingIndices = randomIndices[targetPoseNumber:]

    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    SVD = np.linalg.svd(J)[1]
    newScore = SVD[0]/SVD[-1]
    prevScore = float_info.max

    #print("Initial configurations indices : ")
    #print(CalibrationIndices)
    #print("Initial score : " + str(newScore))

    count = 0
    flagAdd = False
    flagDel = False

    while(newScore < prevScore or count < 20):

        if(newScore == prevScore):
            count += 1
        else:
            count = 0

        prevScore = newScore

        #Perform random swap
        delIndex = np.random.choice(CalibrationIndices)
        addIndex = np.random.choice(RemainingIndices)
        CalibrationIndices = np.delete(CalibrationIndices,np.where(CalibrationIndices == delIndex)[0][0])
        CalibrationIndices = np.append(CalibrationIndices,addIndex)

        CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]
        J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
        SVD = np.linalg.svd(J)[1]
        score = SVD[0]/SVD[-1]

        if(score > prevScore):
            CalibrationIndices = np.delete(CalibrationIndices,-1)
            CalibrationIndices = np.append(CalibrationIndices,delIndex)
            CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]
        else:
            newScore = score

        print("New score : " + str(newScore))

    """
    while(newScore < prevScore or count < 20):

        flagAdd = False
        flagDel = False

        if(newScore == prevScore):
            count += 1
        else:
            count = 0

        prevScore = newScore

        addIndex = np.random.choice(RemainingIndices)
        CalibrationIndices = np.append(CalibrationIndices,addIndex)
        RemainingIndices = np.delete(RemainingIndices,np.where(RemainingIndices == addIndex)[0][0])

        CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]
        J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
        SVD = np.linalg.svd(J)[1]
        score = SVD[0]/SVD[-1]

        if(score > newScore):
            CalibrationIndices = np.delete(CalibrationIndices,-1)
            CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]
        else:
            flagAdd = True
            NotRemovedIndices = np.append(NotRemovedIndices,addIndex)
            newScore = score
            
        if(len(NotRemovedIndices) != 0):
            delIndex = np.random.choice(NotRemovedIndices)
            CalibrationIndices = np.delete(CalibrationIndices,np.where(CalibrationIndices == delIndex)[0][0])
            NotRemovedIndices = np.delete(NotRemovedIndices,np.where(NotRemovedIndices == delIndex)[0][0])

            CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]
            J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
            SVD = np.linalg.svd(J)[1]
            score = SVD[0]/SVD[-1]

            if(score > newScore):
                CalibrationIndices = np.append(CalibrationIndices,delIndex)
                CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]  
            else:
                flagDel = True
                newScore = score

        #if(flagAdd and flagDel):
            #print("Exchanging pose " + str(delIndex) + " with pose " + str(addIndex))
        #elif(flagAdd and not flagDel):
            #print("Adding pose " + str(addIndex))
        #elif(flagDel and not flagAdd):
            #print("Deleting pose " + str(delIndex))
        #else:
            #print("No changes")
            #continue

        #print("New configurations indices : ")
        #print(CalibrationIndices)
        print("New score : " + str(newScore))
    """

    print("Done !")

    print("Saving optimized calibration configurations...")
    with open("CalibrationConfigurations_" + argv[1] + "_" + argv[2] + "_greedy.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.flatten().tolist()}, file)
    print("Done !")

flag = input("Genetic algorithm ? y/n")

if(flag == "y" or flag == "Y"):

    ### GENETIC ALGORITHM

    print("Genetic algorithm...")

    counter = 0
    maxIterations = 20
    num_parents_mating = 4
    maxCounter = maxIterations*num_parents_mating + N
    print("Maximum number of score function calls : " + str(maxCounter))

    def fitnessFunction(solution,index):
        solution = solution.reshape((-1,JOINTS_NUMBER))
        #print(solution)
        J = np.vstack([Jfunc(parameters0,q) for q in solution])
        SVD = np.linalg.svd(J)[1]
        score = SVD[-1]/SVD[0]

        global counter
        counter+=1
        print("score " + str(counter) + "/" + str(maxCounter) + " = " + str(1/score))

        return(score)

    instanceGA = GA(num_generations=maxIterations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitnessFunction,
                    sol_per_pop=N,
                    num_genes=JOINTS_NUMBER*targetPoseNumber,
                    gene_type=float,
                    gene_space=np.hstack([{'low':limits[0],'high':limits[1]} for limits in zip(jointsLowerLimits,jointsUpperLimits)]*targetPoseNumber),
                    parent_selection_type="sss",
                    keep_parents=1,
                    crossover_type="single_point",
                    mutation_type="random",
                    mutation_percent_genes=10)

    instanceGA.run()
    CalibrationConfigurations = instanceGA.best_solution()[0].reshape((-1,JOINTS_NUMBER))
    #print("Final calibration configurations : ")
    #print(CalibrationConfigurations)

    print("Done !")

    print("Saving optimized calibration configurations...")
    with open("CalibrationConfigurations_" + argv[1] + "_" + argv[2] + "_genetic.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.flatten().tolist()}, file)
    print("Done !")

flag = input("Differential evolution algorithm ? y/n")

if(flag == "y" or flag == "Y"):

    ### DIFFERENTIAL EVOLUTION ALGORITHM

    print("Differential evolution algorithm...")

    counter = 0
    maxIterations = 5
    maxCounter = (maxIterations + 1)*JOINTS_NUMBER*targetPoseNumber*N
    print("Maximum number of score function calls : " + str(maxCounter))

    def func(x):
        x = x.reshape((-1,JOINTS_NUMBER))
        #print(x)
        J = np.vstack([Jfunc(parameters0,q) for q in x])
        SVD = np.linalg.svd(J)[1]
        score = SVD[0]/SVD[-1]

        global counter
        counter+=1
        print("score " + str(counter) + "/" + str(maxCounter) + " = " + str(score))

        return(score)

    result = differential_evolution(func = func,
                                    bounds = np.vstack([np.vstack((jointsLowerLimits,jointsUpperLimits)).T]*targetPoseNumber),
                                    popsize = N,
                                    maxiter = maxIterations,
                                    polish = False,
                                    workers = -1)

    CalibrationConfigurations = result.x.reshape((-1,JOINTS_NUMBER))
    #print("Final calibration configurations : ")
    #print(CalibrationConfigurations)

    print("Done !")

    print("Saving optimized calibration configurations...")
    with open("CalibrationConfigurations_" + argv[1] + "_" + argv[2] + "_diffevol.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.flatten().tolist()}, file)
    print("Done !")
    