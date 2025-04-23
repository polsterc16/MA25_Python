# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:19:36 2025

@author: Admin

https://www.geeksforgeeks.org/pytorch-learn-with-examples/
    Optimizing Model Training with PyTorch Datasets
        1. Efficient Data Handling with Datasets and DataLoaders

Generate own Dataset for even distribution of points inside and
    outside of the unitcircle (radius from 0.0 .. 2.0)
"""
#%% IMPORTS
import torch
from torch.utils.data import Dataset, DataLoader

class DataSetUnitCircle2(Dataset):
    def __init__(self, size=1000):
        self.data = torch.zeros((2,size), dtype=torch.float)
        self.labels = torch.zeros((size), dtype=torch.float)
        
        args = torch.rand((2,size), dtype=torch.float)
        args[0] *= 2
        args[1] *= 2*torch.pi
        
        self.data[0] = args[0] * torch.cos( args[1] )
        self.data[1] = args[0] * torch.sin( args[1] )
        
        self.labels = (args[0]<1).int()
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#%%
if __name__ == "__main__":
    dataset = DataSetUnitCircle2(4)
    data = dataset.data
    labels = dataset.labels
    
    pass