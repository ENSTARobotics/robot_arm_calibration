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

### SORTING FUNCTION

def sortConfigurations(configurations):
    if(np.shape(configurations)[1] == 1):
        return(np.sort(configurations))
    else:
        sortedConfigurations = configurations[np.argsort(configurations[:,0])]
        uniqueValues = np.unique(sortedConfigurations[:,0])

        previousIndex = 0
        nextIndex = previousIndex
        output = np.empty((0,len(sortedConfigurations[0])-1))

        for value in uniqueValues:
            nextIndex = np.where(sortedConfigurations[:,0] == value)[0][-1] + 1
            output = np.vstack((output,sortConfigurations(sortedConfigurations[previousIndex:nextIndex,1:])))
            previousIndex = nextIndex

        return(np.hstack((np.array([sortedConfigurations[:,0]]).T,output)))

### RETREIVE NOMINAL PARAMETERS
TBase0 = TBaseN

TTool0 = np.empty((len(TToolN_marker),4,4))
for j in range(len(TToolN_marker)):
    TTool0[j] = np.dot(TFlangeN,TToolN_marker[j])

PBase0 = TBase0[:3,-1]
RBase0 = R.from_matrix(TBase0[:3,:3])
PTool0 = TTool0[:,:3,-1]
pi0 = pi0

parameters0 = np.hstack((PBase0,R.as_euler(RBase0,"xyz"),pi0,PTool0.flatten()))

def random(sampledConfigurations,targetConfigurationsNumber,callsNumber):
    print("Random selection...")

    randomIndices = np.random.permutation(len(sampledConfigurations))
    CalibrationIndices = randomIndices[:targetConfigurationsNumber]
    CalibrationConfigurations = sampledConfigurations[np.sort(CalibrationIndices)]

    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    SVD = np.linalg.svd(J)[1]
    score = SVD[0]/SVD[-1]

    print("score 1/1 = " + str(score))

    print("Done !")
    return(CalibrationConfigurations)

def naive(sampledConfigurations,targetConfigurationsNumber,callsNumber):
    print("Naive selection...")

    naiveN = int(np.round(targetConfigurationsNumber/JOINTS_NUMBER))
    CalibrationConfigurations = np.vstack([(jointsUpperLimits + jointsLowerLimits)/2]*(naiveN*JOINTS_NUMBER))

    for i in range(JOINTS_NUMBER):
        for j in range(naiveN):
            CalibrationConfigurations[i*naiveN + j][i] =  jointsLowerLimits[i] + (jointsUpperLimits[i] - jointsLowerLimits[i])*j/(naiveN-1)

    print("Done !")
    return(CalibrationConfigurations)

def greedy(sampledConfigurations,targetConfigurationsNumber,callsNumber):
    print("Greedy algorithm...")
    print("Maximum number of score function calls : " + str(callsNumber))

    randomIndices = np.random.permutation(len(sampledConfigurations))
    CalibrationIndices = randomIndices[:targetConfigurationsNumber]
    RemainingIndices = randomIndices[targetConfigurationsNumber:]

    CalibrationConfigurations = sampledConfigurations[CalibrationIndices]
    J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
    SVD = np.linalg.svd(J)[1]
    newScore = SVD[0]/SVD[-1]
    prevScore = sys.float_info.max

    print("score " + str(1) + "/" + str(callsNumber) + " = " + str(newScore))

    for i in range(callsNumber - 1):

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

        print("score " + str(i+2) + "/" + str(callsNumber) + " = " + str(newScore))

    CalibrationConfigurations = sampledConfigurations[np.sort(CalibrationIndices)]

    print("Done !")
    return(CalibrationConfigurations)

def genetic(sampledConfigurations,targetConfigurationsNumber,callsNumber):
    print("Genetic algorithm...")

    global counter
    counter = 0

    sol_per_pop = 5*int(np.round(targetConfigurationsNumber*0.1/5))

    product = callsNumber - sol_per_pop
    decompositions = list(zip([d for d in range(2,int(np.sqrt(product))) if product % d == 0],[d for d in range(int(np.sqrt(product))+1,product) if product % d == 0][::-1]))

    num_generations = decompositions[(len(decompositions)-1)//2][1]
    num_parents_mating = sol_per_pop - decompositions[(len(decompositions)-1)//2][0]

    maxCounter = num_generations*(sol_per_pop - num_parents_mating) + sol_per_pop
    print("Maximum number of score function calls : " + str(maxCounter))
    print(sol_per_pop)
    print(num_generations)
    print(num_parents_mating)

    def fitnessFunction(solution,index):
        configurations = sampledConfigurations[solution].reshape((-1,JOINTS_NUMBER))
        J = np.vstack([Jfunc(parameters0,q) for q in configurations])
        SVD = np.linalg.svd(J)[1]
        score = SVD[-1]/SVD[0]

        global counter
        counter+=1
        print("score " + str(counter) + "/" + str(maxCounter) + " = " + str(1/score))

        return(score)

    instanceGA = GA(num_generations=num_generations,
                    sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    num_genes=targetConfigurationsNumber,
                    gene_space=np.arange(0,len(sampledConfigurations)),
                    gene_type=int,
                    allow_duplicate_genes=False,
                    fitness_func=fitnessFunction)

    instanceGA.run()
    CalibrationConfigurations = sampledConfigurations[np.sort(instanceGA.best_solution()[0])].reshape((-1,JOINTS_NUMBER))

    print("Done !")
    return(CalibrationConfigurations)
    
def annealing(sampledConfigurations,targetConfigurationsNumber,callsNumber):
    print("Simulated annealing algorithm...")

    global counter
    counter = 0

    print("Maximum number of score function calls : " + str(callsNumber))

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
            score = SVD[0]/SVD[-1]

            global counter
            counter+=1
            print("score " + str(counter) + "/" + str(callsNumber) + " = " + str(score))

            return(score)

    CCP = CalibrationConfigurationsProblem()
    CCP.steps = callsNumber
    CCP.updates = 5
    result,_ = CCP.anneal()

    CalibrationConfigurations = sampledConfigurations[np.sort(result)].reshape((-1,JOINTS_NUMBER))

    print("Done !")
    return(CalibrationConfigurations)

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

callsNumber = 150
try:
    callsNumber = sys.argv[2]
except:
    print("No iterations number given, defaulting to " + str(callsNumber))

targetConfigurationsNumber = int(np.round(2*len(parameters0)*1.0))  #1.1 ?
try:
    selectionMethod = sys.argv[3]
except:
    print("No target configurations number given, defaulting to " + str(targetConfigurationsNumber))

CalibrationConfigurations = sortConfigurations(locals()[selectionMethod.lower()](SampledCalibrationConfigurations,targetConfigurationsNumber,callsNumber))

print("Saving optimized calibration configurations...")
if(not os.path.isdir(os.getcwd() + "/" + selectionMethod.lower() + "/")):
    os.mkdir(os.getcwd() + "/" + selectionMethod.lower() + "/")
with open(os.getcwd() + "/" + selectionMethod.lower() + "/CalibrationConfigurations.yaml",'w') as file:
    yaml.dump({"calibrationConfigurations":CalibrationConfigurations.tolist()}, file)
print("Done !")