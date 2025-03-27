# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:24:52 2025

@author: Admin
"""
#%% IMPORTS
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
# import json
from tqdm import tqdm
# import cv2
# import tensorflow as tf
#%%
path_out_dir = os.path.join("OUTPUT", "04_ann_to_vhdl")
# Check whether the specified path exists or not
isExist = os.path.exists(path_out_dir)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_out_dir)
    print("The new directory is created!"+"\n"+f"{path_out_dir}")

#%% Get Data

file_name = "circle_model_weights.p"
path_dir_03 = os.path.join("OUTPUT", "03_ann")
fpath_input = os.path.join(path_dir_03, file_name)
 
if not os.path.exists(fpath_input):
    raise Exception(f"Path does not exist: \n{fpath_input}")
    pass

weights = pickle.load( open( fpath_input, "rb" ) )

#%%

w1 = np.transpose(weights[0])
b1 = weights[1]
w2 = np.transpose(weights[2])
b2 = weights[3]

w1FP = np.zeros(w1.shape,np.int32)
for i in range(w1.shape[0]):
    for j in range(w1.shape[1]):
        w1FP[i,j] = int(w1[i,j] * 2**8)
        pass
    pass

b1FP = [int(k*2**8) for k in b1]

w2FP = np.zeros(w2.shape,np.int32)
for i in range(w2.shape[0]):
    for j in range(w2.shape[1]):
        w2FP[i,j] = int(w2[i,j] * 2**8)
        pass
    pass

b2FP = [int(k*2**8) for k in b2]



#%%

path_in_dir = os.path.join("INTPUT", "04_ann_to_vhdl")
 
if not os.path.exists(path_in_dir):
    raise Exception(f"Path does not exist: \n{path_in_dir}")
    pass

file_name = "Entity.vhd"
fpath = os.path.join(path_dir_03, file_name)


#%%



#%%



