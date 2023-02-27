#!/usr/bin/python3

#System packages
import sys
import os
import yaml

#Utility modules
import numpy as np

#Comutation modules
from pygad import GA
from simanneal import Annealer

### Import robot and marker parameters
from robot_arm_calibration.ParametersTools import importParameters, getIdentificationParameters
importParameters()

#Custom packages
from robot_arm_calibration.AnalyticGeometricModel import Jfunc

### Sorting function

def sortConfigurations(configurations):
    """! Sorts the configurations according according to incrementally increasing angular values over all joints.

    @param configurations   The angular configurations to be sorted.

    @return  The sorted angular configurations.
    """

    #Single joint case
    if(np.shape(configurations)[1] == 1):
        return(np.sort(configurations))

    #Mutliple joints case
    else:

        #Sort according to the first joint
        sortedConfigurations = configurations[np.argsort(configurations[:,0])]
        uniqueValues = np.unique(sortedConfigurations[:,0])

        previousIndex = 0
        nextIndex = previousIndex
        output = np.empty((0,len(sortedConfigurations[0])-1))

        #Sort according to the next joints, incrementally
        for value in uniqueValues:
            nextIndex = np.where(sortedConfigurations[:,0] == value)[0][-1] + 1
            output = np.vstack((output,sortConfigurations(sortedConfigurations[previousIndex:nextIndex,1:])))
            previousIndex = nextIndex

        return(np.hstack((np.array([sortedConfigurations[:,0]]).T,output)))

### Retreive nominal parameters
parameters0 = getIdentificationParameters(TBaseN,TFlangeN,TToolN_marker,pi0)

### Utility score functions

utilityScores = {"A" : "A-optimality",
                 "D" : "D-optimality",
                 "E" : "E-optimality",
                 "O1" : "O1-optimality",
                 "O2" : "O2-optimality",
                 "O3" : "O3-optimality",
                 "O4" : "O4-optimality",
                 "O5" : "O5-optimality",
                 "A_accuracy" : "Accuracy driven A-optimality"}

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
    return(np.linalg.det(np.dot(J.T,J)))

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
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    measurementsNumber = float(len(TargetConfigurations))
    return(np.power(np.linalg.det(np.dot(J.T,J)),1/measurementsNumber)/measurementsNumber)

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
    return(e_optimality(TargetConfigurations))

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
    """! Computes the accuracy drivent A-optimality score

    @param TargetConfigurations   The selected configurations.

    @return  The computed score.
    """
    J = np.vstack([Jfunc(parameters0,q) for q in TargetConfigurations])
    JStar = np.vstack([Jfunc(parameters0,qStar) for q in TargetConfigurations])
    return(1/np.trace(np.dot(np.dot(JStar,np.linalg.inv(np.dot(J.T,J))),JStar.T)))

### Optimization method

selectionMethods = {"random" : "Random selection",
                 "greedy" : "Greedy exchange algorithm",
                 "genetic" : "Genetic algorithm",
                 "annealing" : "Simulated annealing algorithm"}

def random(sampledConfigurations,targetConfigurationsNumber,maxCallsNumber=150,utilityScore=O1):
    """! Randomly extract a given number of angular configurations 

    @param sampledConfigurations        The initial configurations set.   
    @param targetConfigurationsNumber   The target configurations number.
    @param maxCallsNumber               The maximum number of utility score calls (unused).
    @param utilityScore                 The utility score to use (unused).

    @return  The selected configurations and the corresponding utility score.
    """

    print("Random selection...")

    randomIndices = np.random.permutation(len(sampledConfigurations))
    CalibrationIndices = randomIndices[:targetConfigurationsNumber]
    TargetConfigurations = sampledConfigurations[np.sort(CalibrationIndices)]

    TargetConfigurations = np.random.normal(0.0,1.0,(targetConfigurationsNumber,JOINTS_NUMBER))

    score = utilityScore(TargetConfigurations)
    print("Final score = " + str(score))

    print("Done !")
    return(TargetConfigurations,score)

def greedy(sampledConfigurations,targetConfigurationsNumber,maxCallsNumber=150,utilityScore=O1):
    """! Extract a given number of angular configurations trying to maximize an utility score with the greedy exchange algorithm.

    @param sampledConfigurations      The initial configurations set.   
    @param targetConfigurationsNumber The target configurations number.
    @param maxCallsNumber             The maximum number of utility score calls -default : 150.
    @param utilityScore               The utility score to use - default : O1.

    @return  The selected configurations and the corresponding utility score.
    """

    print("Greedy algorithm...")

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

        if(score > prevScore):
            CalibrationIndices = np.delete(CalibrationIndices,-1)
            CalibrationIndices = np.append(CalibrationIndices,delIndex)
            RemainingIndices = np.delete(RemainingIndices,-1)
            RemainingIndices = np.append(RemainingIndices,addIndex)
        else:
            newScore = score

        print("score " + str(i+2) + "/" + str(maxCallsNumber) + " = " + str(newScore))

    TargetConfigurations = sampledConfigurations[np.sort(CalibrationIndices)]

    score = newScore
    print("Final score = " + str(score))

    print("Done !")
    return(TargetConfigurations,score)

def genetic(sampledConfigurations,targetConfigurationsNumber,maxCallsNumber=150,utilityScore=O1):
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

    sol_per_pop = 5*int(np.round(targetConfigurationsNumber*0.1/5))

    product = maxCallsNumber - sol_per_pop
    decompositions = list(zip([d for d in range(2,int(np.sqrt(product))) if product % d == 0],[d for d in range(int(np.sqrt(product))+1,product) if product % d == 0][::-1]))

    num_generations = decompositions[(len(decompositions)-1)//2][1]
    num_parents_mating = sol_per_pop - decompositions[(len(decompositions)-1)//2][0]

    maxCounter = num_generations*(sol_per_pop - num_parents_mating) + sol_per_pop

    def fitnessFunction(solution,index):
        configurations = sampledConfigurations[solution].reshape((-1,JOINTS_NUMBER))
        score = 1/utilityScore(configurations)

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
    TargetConfigurations = sampledConfigurations[np.sort(instanceGA.best_solution()[0])].reshape((-1,JOINTS_NUMBER))

    score = 1/instanceGA.best_solution()[1]
    print("Final score = " + str(score))

    print("Done !")
    return(TargetConfigurations,score)
    
def annealing(sampledConfigurations,targetConfigurationsNumber,maxCallsNumber=150,utilityScore=O1):
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
            score = utilityScore(configurations)

            global counter
            counter+=1
            print("score " + str(counter) + "/" + str(maxCallsNumber) + " = " + str(score))

            return(score)

    CCP = CalibrationConfigurationsProblem()
    CCP.steps = maxCallsNumber
    CCP.updates = 5
    result,score = CCP.anneal()

    TargetConfigurations = sampledConfigurations[np.sort(result)].reshape((-1,JOINTS_NUMBER))

    print("Final score = " + str(score))

    print("Done !")
    return(TargetConfigurations,score)

### Main loop
if __name__ == "__main__":

    #import time
    #t0 = time.time()

    ### Load sampled configurations

    SampledCalibrationConfigurations = np.empty((0,0))
    print("Loading sampled target configurations...")
    try:
        with open("SampledConfigurations.yaml","r") as file:
                SampledCalibrationConfigurations = np.array(yaml.load(file, Loader=yaml.FullLoader)["sampledConfigurations"]).reshape((-1,JOINTS_NUMBER))
    except:
        print("No valid sampled configurations file ! Are you in a proper calibrated model folder ?")
        exit(1)
    print("Done !")

    selectionMethod = "random"
    try:
        selectionMethod = sys.argv[1]
    except:
        print("No selection method given, defaulting to " + selectionMethod)

    maxCallsNumber = 150
    try:
        maxCallsNumber = int(sys.argv[2])
    except:
        print("No maximum utility function calls number given, defaulting to " + str(maxCallsNumber))

    targetConfigurationsNumber = int(np.round(2*len(parameters0)*1.0))  #1.1 ?
    try:
        targetConfigurationsNumber = int(sys.argv[3])
    except:
        print("No target configurations number given, defaulting to twice as much as parameters to identify " + str(targetConfigurationsNumber))

    utilityScore = "O1"
    try:
        utilityScore = sys.argv[4]
    except:
        print("No utility score given, defaulting to " + utilityScores[utilityScore])

    TargetConfigurations,score = sortConfigurations(locals()[selectionMethods[selectionMethod]](SampledCalibrationConfigurations,targetConfigurationsNumber,maxCallsNumber,locals()[utilityScores[utilityScore]]))

    print("Saving optimized target configurations...")

    saveDirectory = os.getcwd() + "/" + utilityScore.lower()  + "/" + selectionMethod.lower() + "/"

    if(not os.path.isdir(saveDirectory)):
        os.mkdir(saveDirectory)
    with open(saveDirectory + "/TargetConfigurations.yaml",'w') as file:
        yaml.dump({"targetConfigurations":TargetConfigurations.tolist(),"score":score}, file)
    print("Done !")

    #print("-- Optimal configurations computed in " + str(time.time() - t0) + " seconds --")