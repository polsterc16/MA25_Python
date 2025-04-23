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

from tqdm import tqdm

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


#%% Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

#%% Hyperparameters
input_size = 28*28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

#%% Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=True)

#%% Initialize Network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#%% Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#%% Train Network

for epoch in range(num_epochs):
    print(f"Training for epoch {epoch+1} / {num_epochs}")
    for data, targets in tqdm(train_loader):  # data = image, targets = correct label
        # get data to cuda, if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # data.shape = 64,1,28,28
        
        #unroll data into long vector (keep batch dimension)
        data = data.reshape(data.shape[0], -1)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        # backward
        optimizer.zero_grad() # reset the optimizer for this batch
        loss.backward()
        
        # grad descend OR adam step
        optimizer.step()
        

#%% Check accuracy

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            # get data to device
            x = x.to(device=device)
            y = y.to(device=device)
            
            #unroll  
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            
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