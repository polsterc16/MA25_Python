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


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#%% Network
# https://pytorch.org/docs/main/nn.html#loss-functions
# https://pytorch.org/docs/main/nn.html#non-linear-activations-weighted-sum-nonlinearity

# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

class NN(nn.Module): # inherit from nn.Module

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(NN, self).__init__()
        # Initialize the modules we need to build the network
        # self.linear1 = nn.Linear(num_inputs, num_hidden)
        # self.linear2 = nn.Linear(num_hidden, num_outputs)
        # # self.act_relu = F.relu
        # self.act_relu = nn.ReLU()
        # # self.act_hsigm = F.hardsigmoid
        # self.act_hsigm = nn.Hardsigmoid()
        
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


# save class
# torch.save(NN, "OUTPUT/03c_ann_pytorch_class.tar")

#%% Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )



#%% Hyperparameters
num_inputs = 2
num_hidden = 8
num_outputs = 1

batch_size = 64
num_epochs = 5

#%% Initialize Network

model = NN(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs).to(device)
# Printing a module shows all its submodules
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
def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train()
    
    # Training loop
    # for epoch in tqdm(range(num_epochs)):
    for epoch in (range(num_epochs)):
        # print(f"Epoch {epoch+1} / {num_epochs}")
        for data_inputs, data_labels in tqdm(data_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs:02d}"):
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
#%%

train_model(model, optimizer, train_data_loader, loss_module, num_epochs=num_epochs)
#%% Save Model
state_dict = model.state_dict()

# # torch.save(object, filename). For the filename, any extension can be used
# torch.save(state_dict, "OUTPUT/03c_ann_pytorch.tar")

# # save whole model
# torch.save(model, "OUTPUT/03c_ann_pytorch_model.tar")


# ALTERNATIVE: https://pytorch.org/tutorials/beginner/saving_loading_models.html
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('03c_ann_pytorch_model.pt') # Save

# example_inputs = (torch.randn(2),)
# onnx_program = torch.onnx.export(model,example_inputs, '03c_ann_pytorch_model.onnx')

#%% Load Model

# # Load state dict from the disk 
# state_dict  = torch.load("OUTPUT/03c_ann_pytorch.tar")

# # Create a new model and load the state
# model = NN(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs)
# model.load_state_dict(state_dict)

# ALTERNATIVE: https://pytorch.org/tutorials/beginner/saving_loading_models.html

model = torch.jit.load('03c_ann_pytorch_model.tar')
model.eval()

#%% Get Evaluation Dataset
test_dataset = CircleDataset(size=1e4)
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
            
            # preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1

            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print(f"\nAccuracy of the model: {100.0*acc:4.2f}%")



#%%

eval_model(model, test_data_loader)

#%% Show region

@torch.no_grad()
def visualize_classification(model, data, label):
    model.eval() # Set model to eval mode
    
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    
    fig = plt.figure(figsize=(4,4))
    plt.scatter(data_0[:,0], data_0[:,1], marker="^", label="Outside")
    plt.scatter(data_1[:,0], data_1[:,1], marker="o", label="Inside")
    plt.title("Unit Circle Evaluation")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$x$")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    
    return fig

@torch.no_grad()
def visualize_classification_2(model, data, label):
    model.eval() # Set model to eval mode
    
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    
    # fig = plt.figure(figsize=(4,4))
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(
        data_0[:,0], data_0[:,1], marker="^", linewidths=0.5, facecolors='none', edgecolors="k", label="Outside")
    ax.scatter(
        data_1[:,0], data_1[:,1], marker="o", linewidths=0.5, facecolors='none', edgecolors="k", label="Inside")
    
    ax.set_title("Unit Circle Evaluation")
    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$x$")
    # plt.title("Dataset samples")
    # plt.ylabel(r"$y$")
    # plt.xlabel(r"$x$")
    
    
    
    ax.legend()
    ax.axis("equal")
    ax.grid(True)
    
    
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1))
    
    
    model.to(device)
    
    c0 = torch.Tensor(mpl.colors.to_rgba("C0")).to(device)
    c1 = torch.Tensor(mpl.colors.to_rgba("C1")).to(device)
    
    x1 = torch.arange(-2, 2, step=0.01, device=device)
    x2 = torch.arange(-2, 2, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds = model(model_inputs)
    
    output_image = (1 - preds) * c0[None,None] + preds * c1[None,None]
    output_image = output_image.cpu().numpy()
    
    
    
    ax.imshow(output_image, origin='lower', extent=(-2, 2, -2, 2))
    
    
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    # plt.xlim([-2, 2])
    # plt.ylim([-2, 2])
    
    # fig.tight_layout()
    return fig,ax


#%%
# visual_dataset = CircleDataset(size=1e3)


# plt.close("all")
# _ = visualize_classification(model, visual_dataset.data, visual_dataset.label)
# plt.show()



visual_dataset = CircleDataset(size=1e2)


plt.close("all")
fig,ax = visualize_classification_2(model, visual_dataset.data, visual_dataset.label)
plt.show()

#%%
# raise Exception("end")

#%%

# with open("OUTPUT/temp.csv", "w") as f:
    
#     for x,y in train_dataset:
#         # print(x)
#         # print(y)
#         # raise Exception()
#         f.write(f"{x[0]:.3f};{x[1]:.3f};{y:.3f}\n")


