#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

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

#ToDO: Read in inputs and ouputs from file

file = pd.read_csv("wdbc.csv")
df = pd.DataFrame(file)
df = df.sample(frac=1,random_state=params.seed)

length = len(df.index) #568
length = math.floor(length*(3/4)) #426

train = df[:length] #0-426
train_length = len(train.index)
print("Total Train size: ",train_length)
#print(train)

test = df[train_length:len(df.index)] #426-568 (142)
test_length = len(test.index)
print("Total Test size: ",test_length)

train = train.to_numpy()
test = test.to_numpy()


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

def evalSymbReg(individual):
    func = toolbox.compile(expr=individual)

    hits = 0
    for x, y in zip(train[:,2:],train[:,1:2]):
        if (func(*x) >= 0.0) and(y[0] == "M"):
            hits = hits + 1
        elif (func(*x) < 0.0 and y[0] == "B"):
            hits = hits + 1
    return hits,

toolbox.register("evaluate", evalSymbReg) # In list
toolbox.register("select", tools.selTournament, tournsize=params.tournamentSize)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params.maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params.maxDepth))

def run():

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

    pop, log = algorithms.eaSimple(pop, toolbox, params.crossoverRate, params.mutateRate, params.numGenerations, stats=mstats,
                                halloffame=hof, verbose=True)       
    # print log and best
    best = tools.selBest(pop, k=1)
    best = best[0]
    print("Best Individual: ")
    print(best)
    tree = gp.compile(best,pset)
    #print(in_[1:2])


    #p = in_test[2:]
    #print(tree(x))

    print("Running Tests: ")
    hits_malig = 0
    false_pos = 0
    true_pos = 0 
    true_neg = 0 
    false_neg = 0
    for x, y in zip(test[:,2:],test[:,1:2]):
        val = tree(*x)
        type = y[0]
        
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

    print("True Pos: ",true_pos," True Neg: ",true_neg,
    " False Pos: ",false_pos," False Neg: ",false_neg)

if __name__ == "__main__":
    run()