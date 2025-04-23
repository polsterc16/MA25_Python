# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 10:23:53 2025

@author: Admin

https://youtu.be/x9JiIFvlUwk
"""
#%% IMPORTS
# import os

import torch
#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device=device)

#%%
# initialisation of tensors
x = torch.empty((3,3))
y = torch.zeros((3,3))
z = torch.ones((3,3))

x = torch.rand((3,3))
print(x)

x = torch.eye(3,3)
print(x)

x = torch.arange(start=0, end=5, step=1)
print(x)

x = torch.linspace(0.1, 1, steps=10)
print(x)

# specify kind of distribution
x = torch.empty((1,5)).normal_(mean=0, std=1)
print(x)
x = torch.empty((1,5)).uniform_(0.5,3)
print(x)

# diagonal matrix from vector
x = torch.diag(torch.ones(3))
print(x)


#%%
# initialise and convert
x = torch.arange(4)
print(x.bool())
print(x.short())
print(x.long())

print(x.half())
print(x.float())
print(x.double())


#%%
# array to tensor conversion
import numpy as np

a = np.random.random((5,5))
print(a)
x = torch.from_numpy(a)
print(x)
b = x.numpy()
print(b)

#%%

# Maths and Comparison

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = torch.empty(3)
torch.add(x,y, out=z1)
print(z1)

z2 = torch.add(x,y)
z3 = x + y

# Subtraction
z = x-y

# Division
z = torch.true_divide(x, y)

# Inplace
t = torch.zeros(3)
t.add_(x)   # inplace
print(t)
# FUNC_ is indicator that function is IN PLACE

t = torch.zeros(3)
t += x      # inplace

t = torch.zeros(3)
t = t + x   # NOT inplace

# Comparison
z = x > 0
print(z)
z = x < 0
print(z)


# Element wise Multiplication
z = x * y
print(z)


# Dot Product
z = torch.dot(x,y)
print(z)


# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))

x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

z = x.pow(2)
z = x ** 2

# Matrix exponentiation
matrix_exp = torch.rand(5,5)
z = matrix_exp.matrix_power(3)  # M * M * M
print(z)

# Batch matrix multiplication
batch=32
n=10
m=20
p=30

t1 = torch.rand((batch, n,m))
t2 = torch.rand((batch, m,p))
t_bmm = torch.bmm(t1, t2) # shape: (batch, n, p)



# Broadcasting

x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

# broadcasting = autom expansion of dimension to match in shape
z = x1 - x2
z = x1 ** x2


# other useful operations
sum_x = torch.sum(x, dim=0)
val, idx = torch.max(x, dim=0)
val, idx = torch.min(x, dim=0)

abs_x = torch.sum(x)

mean_x = torch.mean(x.float(), dim=0)

# equal?
z=torch.eq(x,y)

# sorting
sorted_y, indices = torch.sort(y, dim=0, descending=False)

# define min and/or max values for tensor
z = torch.clamp(x, min=0, max=10)

# bool checks 
x = torch.tensor( [1,0,0,1,1,1], dtype=torch.bool )

z = torch.any(x) # check if ANY is TRUE
z = torch.all(x) # check if ALL are TRUE

#%%

# Tensor indexing

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

# get (shape of) features of first example
print( x[0].shape )
print( x[0,:].shape )

# get (shape of) first feature of all example
print( x[:,0].shape )

# get first 10 features of the 3rd example
print( x[2, 0:10])

# fancy indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])


x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols])


# advanced indixing
x = torch.arange(10)
print( x[(x<2) | (x>8)] ) # or indexing
print( x[(x<2) & (x>8)] ) # and indexing

print( x[x.remainder(2)==0])


# USEFUL OPERATIONS
print(torch.where(x>5, -x, x*2)) # where (CONDITION do ARG_TRUE, else do ARG_FALSE)

print(torch.tensor([0,0,1,2,2,3,4]).unique()) # returns tensor with unique elements

print(x.ndimension()) # return NUmber of dimensions of object

print(x.numel()) # returns count of elements in x

#%%

# Tensor reshaping

x = torch.arange(9)

x_3x3 = x.view(3,3)     # acts on continuous tensors (contiguous in memory)
x_3x3 = x.reshape(3,3)  # can act on non-contiguous memory
print(x_3x3)

y = x_3x3.t() # transpose
print(y)
# y is now not continuous in memory
# reshape is safer


# concatination of tensors
x1 = torch.rand((2,5))
x2 = torch.rand((2,5))

y1 = torch.cat((x1,x2), dim = 0)
y2 = torch.cat((x1,x2), dim = 1)

print(y1.shape)
print(y2.shape)

# unrolling of tensor
z = x1.view(-1)
print(z.shape)

# unrolling with batch
batch=64
x = torch.rand((batch,2,5))
z = x.view(batch, -1)
print(z.shape)

# switching axes (with batch)
z=x.permute(0,2,1)  # dim 0 stays at pos 0,
                    # dim 2 goes to pos 1
                    # dim 1 goes to pos 2
print(z.shape)

# .t() is special case of .permute(...)


# UNSQUEEZE
x = torch.arange(10)
print(x.shape)

print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)


# SQUEEZE
x = torch.arange(10).unsqueeze(0).unsqueeze(1) # shape: 1x1x10
print(x.squeeze(0).shape)
print(x.squeeze(1).shape)
