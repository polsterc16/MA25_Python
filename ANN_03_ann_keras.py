# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:32:36 2025

@author: Admin
"""
#%% IMPORTS

import os

import numpy as np

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
#%% ANN

model = Sequential() 
L1 = 8

model.add(Dense( L1, activation = 'relu', input_shape = (2,) )) 

model.add(Dense(1, activation = 'hard_sigmoid'))
# model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])

#%% Get Data
# https://stackoverflow.com/questions/70230687/how-keras-utils-sequence-works/70319612#70319612

class DatasetCircle(keras.utils.Sequence):

    def __init__(self, length, batch_size=32, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        # self.x = x_in
        # self.y = y_in
        self.datalen = length
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.__gen_data()

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch
    
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __gen_data(self):
        # self.x = np.zeros((self.datalen,2))
        # self.y = np.zeros((self.datalen,))
        
        array_radius = np.random.rand(self.datalen)*2
        array_angle = np.random.rand(self.datalen)*2*np.pi
        
        self.x = np.array([array_radius*np.cos(array_angle), array_radius*np.sin(array_angle)])
        self.x = np.transpose(self.x)
        self.y = array_radius<=1
        
        pass

#%%

N = 1e5

training_generator   = DatasetCircle(int(N / 32)*32, batch_size=32)
validation_generator = DatasetCircle(int(N/2/32)*32, batch_size=32)

#%%
print("-- model training")
NUM_EPOCHS = 5

# model.fit(X_train, y_train, epochs=5, sample_weight=w_train)
model.fit_generator(generator=training_generator, validation_data=validation_generator, 
                    epochs=NUM_EPOCHS)

weights = model.get_weights()

print("-- training done")

#%% store model


fpath = os.path.join(path_out_dir, "circle_model.keras")
model.save(fpath, True, True)

raise Exception("End")

#%%
fpath = os.path.join(path_out_dir, "circle_model.keras")

model = keras.models.load_model(fpath)



