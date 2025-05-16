# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:25:14 2025

@author: Admin
"""
#%% IMPORTS

import numpy as np
import keras

#%%

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

z = DatasetCircle(100)
# print( z.__getitem__(0))

x = z.x
y = z.y