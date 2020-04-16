''' Demonstrates usage of the EvoVec module and construction of a vector-weighted MLP in PyTorch from a configuration '''

import torch
import torch.nn as nn
import VectorNet as vn
import EvoVec as ev
import numpy as np

class VectorNet(nn.Module):
    
    def __init__(self, cfg, in_size, out_size):
        super(VectorNet, self).__init__()
        self.cfg = cfg
        self.expands = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.aggregates = nn.ModuleList()
        self.widths = [in_size, 64, 32, 32, 16, out_size]
        height = 1
        for i in range(5):
            l = self.cfg[i]
            
            # expansion
            if l.e:
                height = l.e.size
                self.expands.append(vn.Expand(height, eval(expargs[l.e.fct], {'height':height, 'torch':torch})))
            else:
                self.expands.append(None)
            
            # layer
            if height != 1:
                self.layers.append(vn.LinearVector(self.widths[i], self.widths[i+1], height=height))
            else:
                self.layers.append(nn.Linear(self.widths[i], self.widths[i+1]))
                
            # aggregation
            if l.a:
                height = 1
                self.aggregates.append(vn.Aggregate(eval(aggargs[l.a.fct])))
            else:
                self.aggregates.append(None)

        self.dropout = nn.Dropout(0.25)
        
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for i in range(5):
            if self.expands[i]:
                x = self.expands[i](x)
            x = self.dropout(self.layers[i](x).clamp(min=0)) if i < 5 - 1 else self.layers[i](x)
            if self.aggregates[i]:
                x = self.aggregates[i](x)
        return x
            
expargs = {'ID': '[lambda x: x] * height',
           'NID': '[lambda x: x / torch.tensor(height).pow(1./2)] * height',
           'F': '[lambda x: x] * (height - 2) + [lambda x: (x+1).sqrt(), lambda x: (x+1).log()]'}
    
aggargs = {'MEAN': 'lambda x: x.mean(dim=-1)',
           'L2': 'lambda x: x.norm(dim=-1)',
           'MAX': 'lambda x: x.max(dim=-1)[0]',
           'MED': 'lambda x: x.median(dim=-1)[0]'}
    
nets = ev.EvoPop(5, 20, extra_fcts=True)

print('=== INITIAL POPULATION ===')
for ind, i in enumerate(nets.pop):
    print(ind + 1, i)

# model based on the first configuration in the population
m = VectorNet(nets.pop[0].config, 784, 10)

# generate random fitness scores
fitness = np.random.uniform(0,1, size=20)

nets.evolve(fitness)

print('=== NEXT GENERATION ===')
for ind, i in enumerate(nets.pop):
    print(ind + 1, i)