''' Demonstrates construction of a vector-weighted MLP with the VectorNet module '''

import torch
import torch.nn as nn
import VectorNet as vn

class Model(nn.Module):
    
    def __init__(self, in_size, out_size, height, exp_fct, agg_fct):
        super(Model, self).__init__()
        self.height = height
        self.expand = vn.Expand(self.height, exp_fct)
        self.linear1 = vn.LinearVector(in_size, 64, height=self.height)
        self.linear2 = vn.LinearVector(64, out_size, height=self.height)
        self.aggregate = vn.Aggregate(agg_fct)
            
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.expand(x)
        x = self.linear1(x).clamp(min=0)
        x = self.linear2(x)
        x = self.aggregate(x)
        return x

height = 3
e_id = [lambda x: x] * 3
a_mean = lambda x: x.mean(dim=-1)
    
m = Model(784, 10, 3, e_id, a_mean)