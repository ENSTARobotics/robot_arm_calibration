#!/usr/bin/python3

#System packages
from calendar import c
from importlib import import_module
import sys
import os
import yaml
from copy import deepcopy

#Utility modules
import numpy as np
from scipy.spatial.transform import Rotation as R

#Comutation modules
from pygad import GA
from simanneal import Annealer

### GET CONFIG DIRECTORY PATH
configPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/config/"
sys.path.append(configPath)

### RETRIEVE PARAMETERS FROM DIRECTORY NAME
import re
currentParameters = np.array(re.split('/|_',os.getcwd()))
currentParameters = currentParameters[np.where(currentParameters == "calibration")[0][0] + 1:]

### IMPORT ROBOT PARAMETERS FROM DEDICATED FILE
try:
    module = import_module(currentParameters[0]+"_parameters")

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
    module = import_module(currentParameters[1]+"_parameters")

    if "__all__" in module.__dict__:
        names = module.__dict__["__all__"]
    else:
        names = [x for x in module.__dict__ if not x.startswith("_")]

    globals().update({k: getattr(module, k) for k in names})
except:
    print("INVALID MARKER NAME FOR PARAMETERS IMPORTATION !")
    exit(1)

#Custom packages
from AnalyticGeometricModel import Jfunc

### RETREIVE NOMINAL PARAMETERS
TBase0 = TBaseN

TTool0 = np.empty((len(TToolN_marker),4,4))
for j in range(len(TToolN_marker)):
    TTool0[j] = np.dot(TToolN,TToolN_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

def random(sampledConfigurations,targetConfigurationsNumber):
    print("Random selection...")

    randomIndices = np.random.permutation(len(sampledConfigurations))
    CalibrationIndices = randomIndices[:targetConfigurationsNumber]
    CalibrationConfigurations = sampledConfigurations[np.sort(CalibrationIndices)]

    print("Done !")

    print("Saving optimized calibration configurations...")
    if(not os.path.isdir(os.getcwd() + "/random/")):
        os.mkdir(os.getcwd() + "/random/")
    with open(os.getcwd() + "/random/CalibrationConfigurations.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.tolist()}, file)
    print("Done !")

def naive(sampledConfigurations,targetConfigurationsNumber):
    print("Naive selection...")

    naiveN = int(np.round(targetConfigurationsNumber/JOINTS_NUMBER))
    CalibrationConfigurations = np.vstack([(jointsUpperLimits + jointsLowerLimits)/2]*(naiveN*JOINTS_NUMBER))

    for i in range(JOINTS_NUMBER):
        for j in range(naiveN):
            CalibrationConfigurations[i*naiveN + j][i] =  jointsLowerLimits[i] + (jointsUpperLimits[i] - jointsLowerLimits[i])*j/(naiveN-1)

    print("Done !")

    print("Saving optimized calibration configurations...")
    if(not os.path.isdir(os.getcwd() + "/naive/")):
        os.mkdir(os.getcwd() + "/naive/")
    with open(os.getcwd() + "/naive/CalibrationConfigurations.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.tolist()}, file)
    print("Done !")

def greedy(sampledConfigurations,targetConfigurationsNumber):
    print("Greedy algorithm...")

    randomIndices = np.random.permutation(len(sampledConfigurations))
    CalibrationIndices = randomIndices[:targetConfigurationsNumber]
    RemainingIndices = randomIndices[targetConfigurationsNumber:]

    CalibrationConfigurations = sampledConfigurations[CalibrationIndices]
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    SVD = np.linalg.svd(J)[1]
    newScore = SVD[0]/SVD[-1]
    prevScore = sys.float_info.max

    count = 0

    while(newScore < prevScore or count < 20):

        if(newScore == prevScore):
            count += 1
        else:
            count = 0

        prevScore = newScore

        #Perform random swap
        delIndex = np.random.choice(CalibrationIndices)
        CalibrationIndices = np.delete(CalibrationIndices,np.where(CalibrationIndices == delIndex)[0][0])

        addIndex = np.random.choice(RemainingIndices)
        CalibrationIndices = np.append(CalibrationIndices,addIndex)

        RemainingIndices = np.delete(RemainingIndices,np.where(RemainingIndices == addIndex)[0][0])
        RemainingIndices = np.append(RemainingIndices,delIndex)

        CalibrationConfigurations = sampledConfigurations[CalibrationIndices]
        J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
        SVD = np.linalg.svd(J)[1]
        score = SVD[0]/SVD[-1]

        if(score > prevScore):
            CalibrationIndices = np.delete(CalibrationIndices,-1)
            CalibrationIndices = np.append(CalibrationIndices,delIndex)
            RemainingIndices = np.delete(RemainingIndices,-1)
            RemainingIndices = np.append(RemainingIndices,addIndex)
        else:
            newScore = score

        print("New score : " + str(newScore))

    print("Done !")

    CalibrationConfigurations = sampledConfigurations[np.sort(CalibrationIndices)]

    print("Saving optimized calibration configurations...")
    if(not os.path.isdir(os.getcwd() + "/greedy/")):
        os.mkdir(os.getcwd() + "/greedy/")
    with open(os.getcwd() + "/greedy/CalibrationConfigurations.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.tolist()}, file)
    print("Done !")

def genetic(sampledConfigurations,targetConfigurationsNumber):
    print("Genetic algorithm...")

    global counter
    counter = 0
    maxIterations = 20
    num_parents_mating = 4
    maxCounter = maxIterations*num_parents_mating + samplingValue
    print("Maximum number of score function calls : " + str(maxCounter))

    def fitnessFunction(solution,index):
        configurations = sampledConfigurations[solution].reshape((-1,JOINTS_NUMBER))
        J = np.vstack([Jfunc(parameters0,q) for q in configurations])
        SVD = np.linalg.svd(J)[1]
        score = SVD[-1]/SVD[0]

        global counter
        counter+=1
        print("score " + str(counter) + "/" + str(maxCounter) + " = " + str(1/score))

        return(score)

    instanceGA = GA(num_generations=maxIterations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitnessFunction,
                    sol_per_pop=samplingValue,
                    num_genes=targetConfigurationsNumber,
                    gene_type=int,
                    gene_space=np.arange(0,len(sampledConfigurations)),
                    parent_selection_type="sss",
                    keep_parents=1,
                    crossover_type="single_point",
                    mutation_type="random",
                    mutation_percent_genes=10,
                    allow_duplicate_genes=False)

    instanceGA.run()
    CalibrationConfigurations = sampledConfigurations[np.sort(instanceGA.best_solution()[0])].reshape((-1,JOINTS_NUMBER))

    print("Done !")

    print("Saving optimized calibration configurations...")
    if(not os.path.isdir(os.getcwd() + "/genetic/")):
        os.mkdir(os.getcwd() + "/genetic/")
    with open(os.getcwd() + "/genetic/CalibrationConfigurations.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.tolist()}, file)
    print("Done !")
    
def annealing(sampledConfigurations,targetConfigurationsNumber):
    print("Simulated annealing algorithm...")

    class CalibrationConfigurationsProblem(Annealer):
        def __init__(self):
            initialPick = np.random.permutation(len(sampledConfigurations))
            self.state = initialPick[:targetConfigurationsNumber]
            self.heap = initialPick[targetConfigurationsNumber:]
            super().__init__(self.state)

        def move(self):
            deletedIndex = np.random.randint(0,targetConfigurationsNumber)
            deletedItem = self.state[deletedIndex]
            self.state = np.delete(self.state,deletedIndex)
            self.state = np.append(self.state,np.random.choice(self.heap))
            self.heap = np.append(self.heap,deletedItem)
        
        def energy(self):
            configurations = sampledConfigurations[self.state]
            J = np.vstack([Jfunc(parameters0,q) for q in configurations])
            SVD = np.linalg.svd(J)[1]
            return(SVD[0]/SVD[-1])

    CCP = CalibrationConfigurationsProblem()
    CCP.steps = 100
    CCP.updates = 5
    result,_ = CCP.anneal()

    CalibrationConfigurations = sampledConfigurations[np.sort(result)].reshape((-1,JOINTS_NUMBER))

    print("Done !")

    print("Saving optimized calibration configurations...")
    if(not os.path.isdir(os.getcwd() + "/annealing/")):
        os.mkdir(os.getcwd() + "/annealing/")
    with open(os.getcwd() + "/annealing/CalibrationConfigurations.yaml",'w') as file:
        yaml.dump({"calibrationConfigurations":CalibrationConfigurations.tolist()}, file)
    print("Done !")

### LOAD SAMPLED CONFIGURATIONS
samplingMethod = currentParameters[2]
samplingValue = int(currentParameters[3])

if(not os.path.exists("SampledConfigurations.yaml")):
    print("No corresponding sampled configurations found !")
    exit(1)

SampledCalibrationConfigurations = np.empty((0,0))
print("Loading sampled calibration configurations...")
with open("SampledConfigurations.yaml","r") as file:
        SampledCalibrationConfigurations = np.array(yaml.load(file, Loader=yaml.FullLoader)["sampledConfigurations"]).reshape((-1,JOINTS_NUMBER))
print("Done !")

### MAIN LOOP

selectionMethod = "random"
try:
    selectionMethod = sys.argv[1]
except:
    print("No selection method given, defaulting to " + selectionMethod)

targetConfigurationsNumber = int(2*len(parameters0)*(4/3))
try:
    selectionMethod = sys.argv[2]
except:
    print("No target configurations number given, defaulting to " + str(targetConfigurationsNumber))

locals()[selectionMethod.lower()](SampledCalibrationConfigurations,targetConfigurationsNumber)
