# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:44:53 2025

@author: Admin

https://youtu.be/Jy4wM2X21u0
"""
#%% IMPORTS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#%% Create Fully connected Network

class NN(nn.Module): # inherit from nn.Module
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        pass
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

model = NN(28*28,10)
x = torch.rand(64, 28*28) #64 examples
print(model(x).shape)

#%% Set Device

#%% Hyperparameters

#%% Load Data

#%% Initialize Network

#%% Loss and Optimizer

#%% Train Network

#%% Check accuracy


