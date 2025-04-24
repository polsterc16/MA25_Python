# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:44:53 2025

@author: Admin

https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html#Learning-by-example:-Continuous-XOR
"""
#%% IMPORTS
# import os
# import pickle
from tqdm import tqdm
# import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
from DataLoaderCircle import DataSetUnitCircle2


#%% Create Fully connected Network

class NN(nn.Module): # inherit from nn.Module
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # Some init for my module
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, num_classes)
        pass
    
    def forward(self, x):
        # Function for performing the calculation of the module.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


#%%
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


#%% Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

#%% Hyperparameters
input_size = 2
num_classes = 1
learning_rate = 0.001
batch_size = 64
num_epochs = 100
n_samples = 1000

#%% Load Data
X, y = make_circles(n_samples, noise=0.03, random_state=42)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# train_dataset = DataSetUnitCircle2(1e6) # generate dataset if 1e6 circle points
# train_loader = DataLoader(train_dataset, batch_size=batch_size)

# test_dataset = DataSetUnitCircle2(1e5) # generate dataset if 1e6 circle points
# test_loader = DataLoader(test_dataset, batch_size=batch_size)
# Put data to target device

#%% Initialize Network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#%% Loss and Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

#%% Train Network

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(num_epochs):
    print(f"Training for epoch {epoch+1} / {num_epochs}")
    model.train()
    
    # forward
    y_pred = model(X_train).squeeze()
    loss = loss_fn(y_pred, y_train)
        
    # backward
    optimizer.zero_grad() # reset the optimizer for this batch
    loss.backward()
    
    # grad descend OR adam step
    optimizer.step()
    
    ### Testing
    model_0.eval()
    with torch.inference_mode():

#%% Check accuracy

def check_accuracy(model):
    print("Checking accuracy")
    # if loader.dataset.train:
    #     print("Checking accuracy on training data")
    # else:
    #     print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        
        y_pred = model(x)
        _, predictions = y_pred.max(1)
        
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)
        pass
        
        print(f"Got {num_correct} / {num_samples} with accuracy {num_correct/float(num_samples)*100:.2f}")
        pass
    
    model.train()
    pass

#%%

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)