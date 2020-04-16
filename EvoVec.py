''' Manages a population of VectorNet configurations for an evolutionary search '''

import random, copy
import numpy as np

# some constants
SEL_PRESSURE = 2
EXP_PROB = .6
AGG_PROB = .4
VEC_MIN = 2
VEC_MAX = 6
EXP_FCT = ['ID', 'NID', 'F']
AGG_FCT = ['MEAN', 'L2']
FIX_REM_EXP = .5
FIX_REM_AGG = .5
FCT_MUT_PROB = .05
SIZ_MUT_PROB = .05   

class EvoPop(object):
    
    def __init__(self, length, pop_size, extra_fcts=False):
        self.length = length
        self.pop_size = pop_size
        if extra_fcts:
            global AGG_FCT
            AGG_FCT.extend(['MAX', 'MED'])
        self.pop = [EvoConfig(self.length) for i in range(self.pop_size)]
        
    def evolve(self, fitness):
        order = np.argsort(fitness)
        selected = order[(np.random.uniform(size=(2, self.pop_size)) ** SEL_PRESSURE * self.pop_size).astype(int)]
        new_pop = []
        for i in range(self.pop_size):
            new_ind = EvoConfig.crossover(self.pop[selected[0][i]], self.pop[selected[1][i]])
            new_ind.mutate()
            new_pop.append(new_ind)
        self.pop = new_pop
        
class EvoConfig(object):
    
    def __init__(self, length, init=None):
        self.length = length
        self.config = self.new_config(self.length) if not init else self.from_string(length, init)
        
    def new_config(self, length):
        config = [EvoLayer() for i in range(length)]
        expanded = False
        for i in config:
            if not expanded:
                if random.random() < EXP_PROB:
                    i.e = Expansion(rand=True)
                    expanded = True
            if expanded:
                if random.random() < AGG_PROB:
                    i.a = Aggregation(rand=True)
                    expanded = False
        if expanded:
            config[-1].a = Aggregation(rand=True)
        return config
    
    def from_string(self, length, init):
        try:
            comps = [i for i in init.split(' ') if i != '|']
            config = [EvoLayer() for i in range(length)]
            for ind, i in enumerate(config):
                expstr, aggstr = comps[2 * ind], comps[2 * ind + 1]
                if expstr != '--':
                    i.e = Expansion(fct=expstr[expstr.index(',')+1:expstr.index(')')], size=int(expstr[expstr.index('(')+1:expstr.index(',')]))
                if aggstr != '--':
                    i.a = Expansion(fct = aggstr[aggstr.index('(')+1:aggstr.index(')')])
            return config
        except:
            return None
            
    def __str__(self):
        string = ''
        for i in self.config:
            string += '| '
            string += 'E(' + str(i.e.size) + ',' + i.e.fct + ') ' if i.e else '-- '
            string += 'A(' + i.a.fct + ') ' if i.a else '-- '
        string += '|'
        return string
    
    @staticmethod
    def crossover(x, y):
        if (x.length != y.length):
            raise Exception()
        point = random.randint(1, x.length - 1)
        new = EvoConfig(x.length, False)
        new.config = copy.deepcopy(x.config[:point]) + copy.deepcopy(y.config[point:])
        expanded = False
        for ind, i in enumerate(new.config):
            if i.e:
                if expanded:
                    if random.random() < FIX_REM_EXP:
                        i.e = None
                    else:
                        new.config[ind - 1].a = Aggregation(rand=True)
                expanded = True
            if i.a:
                if not expanded:
                    if random.random() < FIX_REM_AGG:
                        i.a = None
                    else:
                        i.e = Expansion(rand=True)
                expanded = False
        if expanded:
            new.config[-1].a = Aggregation(rand=True)
        return new
    
    def mutate(self):
        for i in self.config:
            if i.e:
                if random.random() < SIZ_MUT_PROB:
                    i.e.size = min(max(i.e.size + random.randint(0,1) * 2 - 1, VEC_MIN), VEC_MAX)
                if random.random() < FCT_MUT_PROB:
                    i.e.fct = random.choice(EXP_FCT)
            if i.a:
                if random.random() < FCT_MUT_PROB:
                    i.a.fct = random.choice(AGG_FCT)
        
class EvoLayer(object):
    
    def __init__(self):
        self.e = None
        self.a = None
        
class Expansion(object):
    
    def __init__(self, rand=False, fct=None, size=None):
        self.fct = fct if not rand else random.choice(EXP_FCT)
        self.size = size if not rand else random.randint(VEC_MIN, VEC_MAX)
        
class Aggregation(object):
    
    def __init__(self, rand=False, fct=None):
        self.fct = fct if not rand else random.choice(AGG_FCT)
        