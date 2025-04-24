# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:05:11 2025

@author: Admin
"""
#%% IMPORTS
# import os


import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

#%%

path_out_dir = os.path.join("OUTPUT", "02_gen_circle_data")
# Check whether the specified path exists or not
isExist = os.path.exists(path_out_dir)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_out_dir)
    print("The new directory is created!")

#%% Generate Data
N=1000*1000*4
r_max = 2
# noise_ampl = 0.025


array_radius = np.random.rand(N)*r_max
array_angle = np.random.rand(N)*2*np.pi

array = np.zeros( (N,3) , np.float32)

# array[:, :2] = (np.random.rand(N,2))
array[:, 0] = array_radius * np.cos(array_angle)
array[:, 1] = array_radius * np.sin(array_angle)
# array_noise = (np.random.rand(N)-0.5) * 2 * noise_ampl


# a=array[:, 0]**2
# b=array[:, 1]**2
# c = a+b+array_noise
array[:, 2] = ( array_radius < 1 )

# array[:, 2][array[:, 2]<1] = -1

# array[:, :2] = array[:, :2] + array_noise 

#%% Pickle

# fpath = os.path.join(path, f"circle_{N}.p")
# pickle.dump( array, open( fpath, "wb" ) )



#%%

fpath = os.path.join(path_out_dir, "circle.p")
pickle.dump( array, open( fpath, "wb" ) )

raise Exception("End")