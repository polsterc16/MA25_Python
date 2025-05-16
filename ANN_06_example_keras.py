# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:47:27 2025

@author: Admin
"""
#%% IMPORTS


from keras.models import Sequential 
from keras.layers import Dense

model = Sequential() 

model.add(Dense( units = 4, activation = 'relu', input_shape = (2,) )) 
model.add(Dense( units = 1, activation = 'sigmoid' ))

model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['accuracy'])

#%%
X_train, y_train, X_test, y_test == [...]
model.fit(X_train, y_train, epochs=5, validation_data=(X_test,y_test))