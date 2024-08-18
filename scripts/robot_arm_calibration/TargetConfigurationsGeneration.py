#!/usr/bin/python3

#System packages
import sys
import os
import yaml
import time

#Utility modules
import numpy as np

#Combinatorial optimization packages
from pygad import GA
from simanneal import Annealer

### Import robot and marker parameters
from robot_arm_calibration.ParametersTools import getIdentifiableParametersPartial, getIdentifiableParametersFull

#Custom packages
from robot_arm_calibration.AnalyticGeometricModel import Jfunc, TBaseN, TFlangeN, TToolN, piN, JOINTS_NUMBER, N_EEPOINTS

### Sorting function

def sortConfigurations(configurations):
    """! Sorts the configurations according according to incrementally increasing angular values over all joints.

    @param configurations   The angular configurations to be sorted.

    @return  The sorted angular configurations.
    """
    
    #TODO improve with a traveling salesman solving method !
    return(configurations[np.lexsort(configurations.T[::-1])])  

### Retreive nominal parameters
parameters0 = []
if len(TToolN) > 1:
    parameters0 = getIdentifiableParametersPartial(TBaseN,TFlangeN,TToolN,piN)
else:
    parameters0 = getIdentifiableParametersFull(TBaseN,TFlangeN,TToolN,piN)

### Utility score functions

utilityScores = {"A" : "A-optimality",
                 "D" : "D-optimality",
                 "E" : "E-optimality",
                 "O1" : "O1-optimality",
                 "O2" : "O2-optimality",
                 "O3" : "O3-optimality",
                 "O4" : "O4-optimality",
                 "O5" : "O5-optimality",
                 "A_accuracy" : "Accuracy driven A-optimality",
                 "A_accuracy_max" : "Variant of the accuracy driven A-optimality"}

def A(TargetConfigurations):
    """! Computes the A-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    return(np.trace(np.dot(J.T,J)))

def D(TargetConfigurations):
    """! Computes the D-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    _,logabsdet = np.linalg.slogdet(np.dot(J.T,J))
    return(logabsdet)

def E(TargetConfigurations):
    """! Computes the E-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    _,SVD = np.linalg.svd(np.dot(J.T,J))
    return(SVD[-1])

def O1(TargetConfigurations):
    """! Computes the O1-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    return(D(TargetConfigurations))

def O2(TargetConfigurations):
    """! Computes the O2-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    _,SVD = np.linalg.svd(np.dot(J.T,J))
    return(SVD[-1]/SVD[0])

def O3(TargetConfigurations):
    """! Computes the O3-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    return(E(TargetConfigurations))

def O4(TargetConfigurations):
    """! Computes the O4-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    _,SVD = np.linalg.svd(np.dot(J.T,J))
    return((SVD[-1]**2)/SVD[0])

def O5(TargetConfigurations):
    """! Computes the O5-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    _,SVD = np.linalg.svd(np.dot(J.T,J))
    inverseSVD = 1/SVD
    return(1/np.sum(inverseSVD))

#Interest configurations
#TODO Custom definition !
qStar = np.zeros(JOINTS_NUMBER)

def A_accuracy(TargetConfigurations):
    """! Computes the accuracy driven "A-optimality" score (i.e. Inverted A-optimality weighted with specific target configurations Jacobian identification matrices)

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    JStar = np.vstack([Jfunc(parameters0,qStar) for qStar in TargetConfigurations])
    return(1/np.trace(np.dot(np.dot(JStar,np.linalg.inv(np.dot(J.T,J))),JStar.T)))

def A_accuracy_max(TargetConfigurations):
    """! Computes the variant accuracy driven A-optimality score (i.e. maximum value among inverted A-optimalities weighted with each specific target configuration Jacobian identification matrix separately)

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    JStarAll = [Jfunc(parameters0,qStar) for qStar in TargetConfigurations]
    return(1/max([np.trace(np.dot(np.dot(JStar,np.linalg.inv(np.dot(J.T,J))),JStar.T)) for JStar in JStarAll]))

### Optimization method

selectionMethods = {"random" : "Random selection",
                 "greedy" : "Greedy exchange algorithm",
                 "genetic" : "Genetic algorithm",
                 "annealing" : "Simulated annealing algorithm"}

def random(sampledConfigurations,targetConfigurationsNumber,configurationsSeed=None,maxCallsNumber=150,utilityScore=O1):
    """! Randomly extract a given number of angular configurations 

    @param sampledConfigurations        The initial configurations set.   
    @param targetConfigurationsNumber   The target configurations number.
    @param maxCallsNumber               The maximum number of utility score calls (unused).
    @param utilityScore                 The utility score to use (unused).

    @return  The selected configurations and the corresponding utility score.
    """

    print("Random selection...")

    if(configurationsSeed is not None):
        TargetConfigurations = configurationsSeed

    else:
        randomIndices = np.random.permutation(len(sampledConfigurations))
        CalibrationIndices = randomIndices[:targetConfigurationsNumber]
        TargetConfigurations = sampledConfigurations[np.sort(CalibrationIndices)]

    score = utilityScore(TargetConfigurations)  #No optimization

    print("Done !")
    return(TargetConfigurations,score)

def greedy(sampledConfigurations,targetConfigurationsNumber,configurationsSeed=None,maxCallsNumber=150,utilityScore=O1):
    """! Extract a given number of angular configurations trying to maximize an utility score with the greedy exchange algorithm.

    @param sampledConfigurations      The initial configurations set.   
    @param targetConfigurationsNumber The target configurations number.
    @param maxCallsNumber             The maximum number of utility score calls -default : 150.
    @param utilityScore               The utility score to use - default : O1.

    @return  The selected configurations and the corresponding utility score.
    """

    print("Greedy algorithm...")

    if(configurationsSeed is not None):
        CalibrationIndices = np.where((configurationsSeed[:,None] == sampledConfigurations).all(2).any(0))[0]
        RemainingIndices = np.delete(np.arange(len(sampledConfigurations)),CalibrationIndices)
    else:
        randomIndices = np.random.permutation(len(sampledConfigurations))
        CalibrationIndices = randomIndices[:targetConfigurationsNumber]
        RemainingIndices = randomIndices[targetConfigurationsNumber:]

    TargetConfigurations = sampledConfigurations[CalibrationIndices]
    newScore = utilityScore(TargetConfigurations)
    prevScore = sys.float_info.max

    print("score " + str(1) + "/" + str(maxCallsNumber) + " = " + str(newScore))

    for i in range(maxCallsNumber - 1):

        prevScore = newScore

        #Perform random swap
        delIndex = np.random.choice(CalibrationIndices)
        CalibrationIndices = np.delete(CalibrationIndices,np.where(CalibrationIndices == delIndex)[0][0])

        addIndex = np.random.choice(RemainingIndices)
        CalibrationIndices = np.append(CalibrationIndices,addIndex)

        RemainingIndices = np.delete(RemainingIndices,np.where(RemainingIndices == addIndex)[0][0])
        RemainingIndices = np.append(RemainingIndices,delIndex)

        TargetConfigurations = sampledConfigurations[CalibrationIndices]
        score = utilityScore(TargetConfigurations)

        #Utility score maximization => Reject minimization
        if(score <= prevScore):
            CalibrationIndices = np.delete(CalibrationIndices,-1)
            CalibrationIndices = np.append(CalibrationIndices,delIndex)
            RemainingIndices = np.delete(RemainingIndices,-1)
            RemainingIndices = np.append(RemainingIndices,addIndex)
        else:
            newScore = score

        print("score " + str(i+2) + "/" + str(maxCallsNumber) + " = " + str(newScore))

    TargetConfigurations = sampledConfigurations[np.sort(CalibrationIndices)]

    score = newScore

    print("Done !")
    return(TargetConfigurations,score)

def genetic(sampledConfigurations,targetConfigurationsNumber,configurationsSeed=None,maxCallsNumber=150,utilityScore=O1):
    """! Extract a given number of angular configurations trying to maximize an utility score with the genetic algorithm.

    @param sampledConfigurations      The initial configurations set.   
    @param targetConfigurationsNumber The target configurations number.
    @param maxCallsNumber             The maximum number of utility score calls -default : 150.
    @param utilityScore               The utility score to use - default : O1.

    @return  The selected configurations and the corresponding utility score.
    """

    print("Genetic algorithm...")

    global counter
    counter = 0

    sol_per_pop = 10
    num_generations = int(np.round((maxCallsNumber - sol_per_pop)/(sol_per_pop - 1)))

    maxCounter = num_generations*(sol_per_pop-1) + sol_per_pop

    def fitnessFunction(self,solution,index):
        configurations = sampledConfigurations[solution].reshape((-1,JOINTS_NUMBER))

        #Utility score maximization => Fitness function maximization
        score = utilityScore(configurations)

        global counter
        counter+=1
        print("score " + str(counter) + "/" + str(maxCounter) + " = " + str(score))

        return(score)

    if(configurationsSeed is not None):
        CalibrationIndices = np.where((configurationsSeed[:,None] == sampledConfigurations).all(2).any(0))[0]
    else:
        randomIndices = np.random.permutation(len(sampledConfigurations))
        CalibrationIndices = randomIndices[:targetConfigurationsNumber]

    instanceGA = GA(num_generations=num_generations,
                    sol_per_pop=sol_per_pop,
                    num_parents_mating=int(sol_per_pop/2),
                    num_genes=targetConfigurationsNumber,
                    mutation_percent_genes=int(max(10,1 + 100/targetConfigurationsNumber)),
                    gene_space=np.arange(0,len(sampledConfigurations)),
                    gene_type=int,
                    allow_duplicate_genes=False,
                    fitness_func=fitnessFunction,
                    save_best_solutions=True)

    instanceGA.initial_population[0] = CalibrationIndices

    instanceGA.run()
    TargetConfigurations = sampledConfigurations[instanceGA.best_solutions[-1]].reshape((-1,JOINTS_NUMBER))

    score = instanceGA.best_solutions_fitness[-1]

    print("Done !")
    return(TargetConfigurations,score)
    
def annealing(sampledConfigurations,targetConfigurationsNumber,configurationsSeed=None,maxCallsNumber=150,utilityScore=O1):
    """! Extract a given number of angular configurations trying to maximize an utility score with the simulated annealing algorithm.

    @param sampledConfigurations      The initial configurations set.   
    @param targetConfigurationsNumber The target configurations number.
    @param maxCallsNumber             The maximum number of utility score calls -default : 150.
    @param utilityScore               The utility score to use - default : O1.

    @return  The selected configurations and the corresponding utility score.
    """

    print("Simulated annealing algorithm...")

    global counter
    counter = 0

    class CalibrationConfigurationsProblem(Annealer):
        def __init__(self):

            if(configurationsSeed is not None):
                CalibrationIndices = np.where((configurationsSeed[:,None] == sampledConfigurations).all(2).any(0))[0]
                RemainingIndices = np.delete(np.arange(len(sampledConfigurations)),CalibrationIndices)
            else:
                randomIndices = np.random.permutation(len(sampledConfigurations))
                CalibrationIndices = randomIndices[:targetConfigurationsNumber]
                RemainingIndices = randomIndices[targetConfigurationsNumber:]

            self.state = CalibrationIndices
            self.heap = RemainingIndices
            super().__init__(self.state)

        def move(self):
            deletedIndex = np.random.randint(0,targetConfigurationsNumber)
            deletedItem = self.state[deletedIndex]
            self.state = np.delete(self.state,deletedIndex)
            self.state = np.append(self.state,np.random.choice(self.heap))
            self.heap = np.append(self.heap,deletedItem)
        
        def energy(self):
            configurations = sampledConfigurations[self.state]

            score = utilityScore(configurations)

            global counter
            counter+=1

            print("score " + str(counter) + "/" + str(maxCallsNumber) + " = " + str(score))
            
	        #Utility score maximization => Energy minimization
            return(-score)

    CCP = CalibrationConfigurationsProblem()
    CCP.steps = maxCallsNumber
    CCP.updates = 5
    result,score = CCP.anneal()

    #Utility score maximization => Energy minimization
    score = -score

    TargetConfigurations = sampledConfigurations[np.sort(result)].reshape((-1,JOINTS_NUMBER))

    print("Done !")
    return(TargetConfigurations,score)

### Main loop
if __name__ == "__main__":

    tic = time.perf_counter()
    ### Load sampled configurations
    SampledCalibrationConfigurations = np.empty((0,0))
    print("Loading sampled target configurations...")
    try:
        with open("SampledConfigurations.yaml","r") as file:
                SampledCalibrationConfigurations = np.array(yaml.load(file, Loader=yaml.FullLoader)["sampledConfigurations"]).reshape((-1,JOINTS_NUMBER))
    except:
        print("No valid sampled configurations file ! Are you in a proper calibrated model folder ?")
        sys.exit(-1)
    print("Done !")
    tac = time.perf_counter()

    print("-- Target configurations loaded in " + str(tac - tic) + " seconds --")

    selectionMethod = "random"
    try:
        tmpSelectionMethod = sys.argv[1].lower()
        if(tmpSelectionMethod not in selectionMethods):
            raise ValueError
        selectionMethod = tmpSelectionMethod
    except:
        print("No selection method given, defaulting to " + selectionMethod)

    maxCallsNumber = 150
    try:
        maxCallsNumber = int(sys.argv[2])
    except:
        print("No maximum utility function calls number given, defaulting to " + str(maxCallsNumber))

    #targetConfigurationsNumber = int(np.round(2*len(parameters0)*1.1))  #1.1 ?
    targetConfigurationsNumber = int(2*np.ceil(len(parameters0)/(3*N_EEPOINTS)))
    try:
        targetConfigurationsNumber = int(sys.argv[3])
    except:
        print("No target configurations number given, defaulting to twice the minimal required number = " + str(targetConfigurationsNumber))

    utilityScore = "O1"
    try:
        tmpUtilityScore = sys.argv[4].upper()
        if(tmpUtilityScore not in utilityScores):
            raise ValueError
        utilityScore = tmpUtilityScore
    except:
        print("No utility score given, defaulting to " + utilityScores[utilityScore])

    configurationsSeedPath = ""
    configurationsSeed = None
    try:
        configurationsSeedPath = sys.argv[5]
        with open(configurationsSeedPath,'r') as f:
            configurationsSeed = np.array(yaml.load(f, Loader=yaml.FullLoader)["targetConfigurations"])

        if(len(configurationsSeed) > targetConfigurationsNumber):
            print("Too many configurations in seed, cropping to the right length !")
            configurationsSeed = configurationsSeed[:targetConfigurationsNumber]
        elif(len(configurationsSeed) < targetConfigurationsNumber):
            raise ValueError
    except:
        print("Not enough configurations in given seed, defaulting to random choice")

    redundancy = 1
    try:
        redundancy = int(sys.argv[6])
    except:
        print("Not redundancy factor give, defaulting to " + str(redundancy))

    tic = time.perf_counter()
    #Get and sort target configurations
    TargetConfigurations,score = locals()[selectionMethod](SampledCalibrationConfigurations,targetConfigurationsNumber,configurationsSeed,maxCallsNumber,locals()[utilityScore])
    tac = time.perf_counter()
    
    print("-- Optimal configurations computed in " + str(tac - tic) + " seconds --")
    
    #Adding redundancy...
    TargetConfigurations = np.tile(sortConfigurations(TargetConfigurations),(redundancy,1))
    score = float(locals()[utilityScore](TargetConfigurations))

    print("Final score with redundancy : " + str(score))

    print("Saving optimized target configurations...")
    saveDirectory = os.getcwd() + "/" + utilityScore + "_" + selectionMethod + "_" + str(len(TargetConfigurations)) + "/"

    if(not os.path.isdir(saveDirectory)):
        os.makedirs(saveDirectory)
    with open(saveDirectory + "/TargetConfigurations.yaml",'w') as file:
        yaml.dump({"targetConfigurations":TargetConfigurations.tolist(),"score":score}, file)
    print("Done !")


