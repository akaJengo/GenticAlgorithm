
import pandas as pd

import operator
import math
import random
import numpy
import params

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from functools import reduce
from operator import add, itemgetter

#If you would like to output the testing data to 
#run.csv in the current wroking folder. 
output_to_csv = True

#Reads the csv file and shuffles the data using a set seed.
file = pd.read_csv("wdbc.csv")
df = pd.DataFrame(file)
df = df.sample(frac=1,random_state=params.seed)

#Calculates the length of the training data and splits accordingly
length = len(df.index) #Total length 568
length = math.floor(length*(3/4)) # Split at 426

#Sets the trainging data
train = df[:length] #0-426
train_length = len(train.index)
print("Total Train size: ",train_length)

#Sets the testing data
test = df[train_length:len(df.index)] #426-568 (142)
test_length = len(test.index)
print("Total Test size: ",test_length)
print("Random num: ",params.seed)

train = train.to_numpy()
test = test.to_numpy()

#Creates functions to be used in the genetic program
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", params.terminals)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.addTerminal(-1)
pset.addTerminal(1)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=17)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#This is the evaluation function, simply put, if the 
#guess was correct it increases hits. The point of the
#GP is to maximize our hits on the algorithm. If it is
#Malignant and the evalutation of the tree is greater than
#zero then we increase hits, otherwise if evaluation is 
#negative and Benign, we decrease hits.
def evalSymbReg(individual):
    func = toolbox.compile(expr=individual)

    hits = 0
    for x, y in zip(train[:,2:],train[:,1:2]):
        if (func(*x) >= 0.0) and(y[0] == "M"):
            hits = hits + 1
        elif (func(*x) < 0.0 and y[0] == "B"):
            hits = hits + 1
    return hits,

#Registers the parameters for the genetic program. 
toolbox.register("evaluate", evalSymbReg) # In list
toolbox.register("select", tools.selTournament, tournsize=params.tournamentSize)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params.maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params.maxDepth))

def run():
    #Catalogs the statistical output of the genetic program. 
    random.seed(params.seed)
    pop = toolbox.population(n=params.popSize)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #Runs the genetic program
    pop, log = algorithms.eaSimple(pop, toolbox, params.crossoverRate, params.mutateRate, params.numGenerations, stats=mstats,
                                halloffame=hof, verbose=True) 

    #Next few lines prepare the statistical output and prepare
    #to output into a CSV file called run.csv.

    if output_to_csv == True:
        chapter_keys = log.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in log.chapters.values()]

        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                    in zip(sub_chaper_keys, log.chapters.values())]
        data = numpy.array([[*a, *b] for a, b in zip(*data)])

        columns = reduce(add, [["_".join([x, y]) for y in s] 
                            for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)

        keys = log[0].keys()
        data = [[d[k] for d in log] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        df.to_csv("run.csv")      

    #Selects the best individual from the final generation
    best = tools.selBest(pop, k=1)
    best = best[0]
    print("Best Individual: ")
    print(best)
    tree = gp.compile(best,pset)

    #Using the best selected individual runs the testing data
    #on the best selected tree. Then creates the confusion matrix
    #for this particular run. 
    print("Running Tests: ")
    false_pos = 0
    true_pos = 0 
    true_neg = 0 
    false_neg = 0
    for x, y in zip(test[:,2:],test[:,1:2]):
        val = tree(*x)
        type = y[0]
        
        #If the evaluation is greater than zero and malignant
        #it is marked as a true positive, otherwise it is 
        #benign and greater than zero which is a false positive. 
        #If the value is less than zero and benign it is a true,
        #negative, otherwise it must be a false negative. 
        if val > 0.0:
            if type == 'M':
                true_pos += 1
            else:
                false_pos += 1
        else:
            if type == 'B':
                true_neg += 1
            else:
                false_neg +=1 

    #Output of the confusion matrix 
    print("True Pos: ")
    print(true_pos)
    print("True Neg: ")
    print(true_neg)
    print("False Pos: ")
    print(false_pos)
    print("False Neg: ")
    print(false_neg)

#Main driver
if __name__ == "__main__":
    run()