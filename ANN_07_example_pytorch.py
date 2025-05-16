# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:36:50 2025

@author: Admin
"""
#%% IMPORTS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as data
#%% Network

class NN(nn.Module): # inherit from nn.Module

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(NN, self).__init__()
        
        self.linear_stack = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_outputs),
            nn.Hardsigmoid(),
        )
        pass
    
    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear_stack(x)
        return x


model = NN(num_inputs=2, num_hidden=4, num_outputs=1)
print(model)

#%% Dataset
class CircleDataset(data.Dataset):
    def __init__(self, size=1000):
        super().__init__()
        size = int(size)
        self.size = size
        self.generate_dataset()
        
    
    def generate_dataset(self):
        self.data = torch.zeros((self.size,2), dtype=torch.float32)
        self.label = torch.zeros((self.size), dtype=torch.float32)
        
        args = torch.rand((self.size,2), dtype=torch.float32)
        args[:,0] *= 2            # rand radius 0 to 2
        args[:,1] *= 2*torch.pi   # rand angle 0 to 2*pi
        
        self.data[:,0] = args[:,0] * torch.cos( args[:,1] ) # x pos
        self.data[:,1] = args[:,0] * torch.sin( args[:,1] ) # y pos
        
        self.label = (args[:,0]<1).long()
        
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size
    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        return self.data[idx], self.label[idx]


#%% Get Training Dataset
# data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
train_dataset = CircleDataset(size=1e6)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

#%% Loss and Optimizer
loss_module = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
#%% Train Network
def train_model(model, optimizer, data_loader, loss_module, num_epochs=10):
    model.train() # Set model to train mode
    
    # Training loop
    for epoch in (range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            
            loss = loss_module(preds, data_labels.float())
            
            optimizer.zero_grad() # reset optimizer
            loss.backward()
            
            optimizer.step()