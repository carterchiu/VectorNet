''' PyTorch components for constructing vector-weighted MLPs and CNNs '''

import torch
import torch.nn as nn

class LinearVector(nn.Module):
        
    def __init__(self, in_features, out_features, height):
        super(LinearVector, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.height = height
        self.params = nn.ModuleList(nn.Linear(in_features, out_features) for i in range(self.height))

    def forward(self, input):
        return torch.cat([*[self.params[n](input[:,:,n]).unsqueeze(-1) for n in range(self.height)]], -1)
    
class Conv2dVector(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(Conv2dVector, self).__init__()
        self.height = kwargs['height']
        del kwargs['height']
        self.params = nn.ModuleList(nn.Conv2d(*args, **kwargs) for i in range(self.height))
    
    def forward(self, input):
        return torch.cat([*[self.params[n](input[:,:,:,:,n]).unsqueeze(-1) for n in range(self.height)]], -1)
    
class Pool2dVector(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(Pool2dVector, self).__init__()
        if kwargs['pool'] == 'max':
            self.pool = nn.MaxPool2d(*args)
        elif kwargs['pool'] == 'avg':
            self.pool = nn.AvgPool2d(*args)
        else:
            raise ValueError
        
    def forward(self, input):
        return torch.cat([*[self.pool(input[:,:,:,:,n]).unsqueeze(-1) for n in range(input.shape[-1])]], -1)
    
class BatchNorm1dVector(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(BatchNorm1dVector, self).__init__()
        self.height = kwargs['height']
        del kwargs['height']
        self.params = nn.ModuleList(nn.BatchNorm1d(*args, **kwargs) for i in range(self.height))
        
    def forward(self, input):
        return torch.cat([*[self.params[n](input[:,:,n]).unsqueeze(-1) for n in range(self.height)]], -1)

    
class Expand(nn.Module):
    
    def __init__(self, height, lmb_list=None):
        super(Expand, self).__init__()
        self.height = height
        self.lmb_list = lmb_list
        
    def forward(self, input):
        if self.lmb_list == None:
            return input.unsqueeze(-1).repeat(*([1] * input.dim()), self.height)
        else:
            return torch.cat([*[f(input).unsqueeze(-1) for f in self.lmb_list]], -1)
        
class Aggregate(nn.Module):
    
    def __init__(self, lmb):
        super(Aggregate, self).__init__()
        self.lmb = lmb
        
    def forward(self, input):
        return self.lmb(input)