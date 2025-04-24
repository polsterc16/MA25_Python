# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:09:34 2025

@author: Admin
"""
#%% IMPORTS

import os

import numpy as np
# import matplotlib.pyplot as plt
import pickle
# import json
from tqdm import tqdm
# import cv2

print("numpy version:", np.__version__)


import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#%%
from sklearn.model_selection import train_test_split
# import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout  

import keras

#%%

path_out_dir = os.path.join("OUTPUT", "03_ann")
# Check whether the specified path exists or not
isExist = os.path.exists(path_out_dir)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_out_dir)
    print("The new directory is created!"+"\n"+f"{path_out_dir}")

#%% Get Data

file_name = "circle.p"
path_dir_02 = os.path.join("OUTPUT", "02_gen_circle_data")
fpath_input = os.path.join(path_dir_02, file_name)
 
if not os.path.exists(fpath_input):
    raise Exception(f"Path does not exist: \n{fpath_input}")
    pass

array = pickle.load( open( fpath_input, "rb" ) )
np.random.shuffle(array)
X = array[:,:2]
Y = array[:,2]
del array

#%% ANN
train_ratio = 0.33
idx_train = int(len(Y)*train_ratio)

X_train = X[:idx_train,:]
X_test  = X[idx_train:,:]
y_train = Y[:idx_train]
y_test  = Y[idx_train:]

# X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size=1/3)

model = Sequential() 

L1 = 8
L2 = 2
DO1 = 0.25
DO2 = 0.25

model.add(Dense( L1, activation = 'relu', input_shape = (2,) )) 
# model.add(Dropout(DO1))

# model.add(Dense(L2, activation = 'relu')) 
# model.add(Dropout(DO2)) 

model.add(Dense(1, activation = 'hard_sigmoid'))
# model.add(Dense(1, activation = 'sigmoid'))

# model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
# this example is a binary – classification task

model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
#%%
print("-- model training")
NUM_EPOCHS = 5

# model.fit(X_train, y_train, epochs=5, sample_weight=w_train)
model.fit(X_train, y_train, epochs=NUM_EPOCHS,validation_data=(X_test,y_test))
# model.fit(X_train, y_train, epochs=5,validation_data=(X_test,y_test, w_test), sample_weight=w_train)

weights = model.get_weights()

print("-- training done")
raise Exception("End")
#%% store model
# raise Exception()

fpath = os.path.join(path_out_dir, "circle_model_weights.p")
pickle.dump( weights, open( fpath, "wb" ) )


fpath = os.path.join(path_out_dir, "circle_model_weights.txt")

with open(fpath, "w") as file1:
    # Writing data to a file
    file1.write(str(weights))
    pass

fpath = os.path.join(path_out_dir, "circle_model.keras")
model.save(fpath, True, True)

raise Exception("End")

#%%
fpath = os.path.join(path_out_dir, "circle_model.keras")

model = keras.models.load_model(fpath)




#%%
Nsel = 20

xt2 = X_train[:Nsel,:]
yt2 = y_train[:Nsel]

# xt2 = ( np.random.rand(Nsel,2)-0.5 )*2 *2
# yt2 = np.transpose( np.array( (np.sqrt(np.square(xt2[:,0]) + np.square(xt2[:,1])),) ) )

pred = model.predict( xt2 )

yt2x = np.transpose (np.array((yt2,)))
res = np.concatenate((xt2, pred, yt2x), axis=1)

#%%


pred = model.predict( np.array(((0,0),)) )
