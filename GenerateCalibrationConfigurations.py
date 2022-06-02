from importlib import import_module
from sys import argv, float_info
import os
import numpy as np

from AnalyticGeometricModel import Jfunc
from copy import deepcopy
import yaml

from pygad import GA

from scipy.spatial.transform import Rotation as R

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


### FIRST STEP : GREEDY ALGORITHM

print("Greedy algorithm...")

randomIndices = np.random.permutation(N**JOINTS_NUMBER)

targetPoseNumber = 2*len(parameters0)
CalibrationIndices = randomIndices[:targetPoseNumber]
NotRemovedIndices = deepcopy(CalibrationIndices)
RemainingIndices = randomIndices[targetPoseNumber:]

CalibrationConfigurations = SampledCalibrationConfigurations[CalibrationIndices]
J = np.vstack([Jfunc(parameters0,q) for q in CalibrationConfigurations])
SVD = np.linalg.svd(J)[1]
newScore = SVD[0]/SVD[-1]
prevScore = float_info.max

print("Initial configurations indices : ")
print(CalibrationIndices)
print("Initial score : " + str(newScore))

count = 0
flagAdd = False
flagDel = False

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

    if(flagAdd and flagDel):
        print("Exchanging pose " + str(delIndex) + " with pose " + str(addIndex))
    elif(flagAdd and not flagDel):
        print("Adding pose " + str(addIndex))
    elif(flagDel and not flagAdd):
        print("Deleting pose " + str(delIndex))
    else:
        print("No changes")
        continue

    print("New configurations indices : ")
    print(CalibrationIndices)
    print("New score : " + str(newScore))

print("Done !")

print("Saving optimized calibration configurations...")
with open("CalibrationConfigurations_" + argv[1] + "_" + argv[2] + "_Greedy.yaml",'w') as file:
    yaml.dump({"calibrationConfigurations":CalibrationConfigurations.flatten().tolist()}, file)
print("Done !")

### SECOND STEP : GENETIC ALGORITHM

print("Genetic algorithm...")

def fitnessFunction(solution,index):
    solution = solution.reshape((-1,JOINTS_NUMBER))
    print(solution)
    J = np.vstack([Jfunc(parameters0,q) for q in solution])
    SVD = np.linalg.svd(J)[1]
    score = SVD[-1]/SVD[0]
    print(1/score)
    return(score)

initialPopulation = np.vstack([CalibrationConfigurations.flatten()]*(N//2))

targetPoseNumber = len(CalibrationConfigurations)
randomIndices = np.random.permutation(N**JOINTS_NUMBER)

for i in range(N//2):
    CalibrationConfigurations = SampledCalibrationConfigurations[randomIndices[targetPoseNumber*i:targetPoseNumber*(i+1)]]
    initialPopulation = np.vstack((initialPopulation,CalibrationConfigurations.flatten()))

instanceGA = GA(num_generations=100,
                num_parents_mating=4,
                fitness_func=fitnessFunction,
                initial_population=initialPopulation,
                gene_type=float,
                gene_space=np.vstack([np.linspace(jointsLowerLimits,jointsUpperLimits,N).T]*len(CalibrationConfigurations)),
                parent_selection_type="sss",
                keep_parents=1,
                crossover_type="single_point",
                mutation_type="random",
                mutation_percent_genes=10)

instanceGA.run()
CalibrationConfigurations = instanceGA.best_solution()[0].reshape((-1,JOINTS_NUMBER))
print("Final calibration configurations : ")
print(CalibrationConfigurations)

print("Done !")

print("Saving optimized calibration configurations...")
with open("GACalibrationConfigurations_" + argv[1] + "_" + argv[2] + "Genetic.yaml",'w') as file:
    yaml.dump({"calibrationConfigurations":CalibrationConfigurations.flatten().tolist()}, file)
print("Done !")
