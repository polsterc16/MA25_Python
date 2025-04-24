# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:44:53 2025

@author: Admin

https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html#Learning-by-example:-Continuous-XOR
"""
#%% IMPORTS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as data


from tqdm import tqdm

#%% Network

class NN(nn.Module): # inherit from nn.Module

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(NN, self).__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.act_fn = nn.Tanh()
        pass
    
    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        return x



#%% Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )


#%% Hyperparameters
num_inputs = 2
num_hidden = 4
num_outputs = 1

batch_size = 64
num_epochs = 5

#%% Initialize Network

model = NN(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs).to(device)
# Printing a module shows all its submodules
print(model)

#%% Dataset
class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()
        pass
    
    def generate_continuous_xor(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32) # generate input pairs of 0 and 1
        label = (data.sum(dim=1) == 1).to(torch.long) # xor: sum over inputs == 1
        data += self.std * torch.randn(data.shape) # add gaussian noise to data points
        self.data = data
        self.label = label
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size
    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        return self.data[idx], self.label[idx]

# dataset = XORDataset(size=200)
# print("Size of dataset:", len(dataset))
# print("Data point 0:", dataset[0])

#%% Get Training Dataset
# data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
train_dataset = XORDataset(size=2500)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

#%% Loss and Optimizer
loss_module = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#%% Train Network
def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            # Step 1: Move input data to device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            
            # Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            
            # Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())
            
            # Step 4: Perform backpropagation
            optimizer.zero_grad() # reset optimizer
            loss.backward()
            
            # Step 5: Update the parameters
            optimizer.step()
        
train_model(model, optimizer, train_data_loader, loss_module)
#%% Save Model
state_dict = model.state_dict()

# torch.save(object, filename). For the filename, any extension can be used
torch.save(state_dict, "OUTPUT/test_pytorch_3.tar")

#%% Load Model
state_dict = model.state_dict()

# Load state dict from the disk 
state_dict  = torch.load("OUTPUT/test_pytorch_3.tar")

# Create a new model and load the state
model = NN(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs)
model.load_state_dict(state_dict)

#%% Get Evaluation Dataset
test_dataset = XORDataset(size=500)
# drop_last -> Don't drop the last batch although it is smaller than 128
test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

#%% Evaluate Model

def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.
    
    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:

            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")



#%%

eval_model(model, test_data_loader)