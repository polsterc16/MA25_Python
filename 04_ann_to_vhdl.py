# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:24:52 2025

@author: Admin
"""
#%% IMPORTS
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pickle
# import json
from tqdm import tqdm
# import cv2
import tensorflow as tf
import keras
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

fpath_input = os.path.join(path_dir_03, "circle_model.keras")

model = keras.models.load_model(fpath_input)

weights = model.get_weights()

#%%

ann_conf = model.get_config()

n = ann_conf["name"]
# must be a sequential ANN
assert n[:10] == "sequential"

ann_layers = ann_conf["layers"]


layers = {"num_layers":0}
units_prev = 0
layer_idx = 0

for i,entry in enumerate(ann_layers):
    print("-- Layer", i, entry["class_name"])
    
    if entry["class_name"] == "InputLayer":
        layer_conf = entry["config"]
        layers["input_size"] = layer_conf["batch_input_shape"][1]
        units_prev = layer_conf["batch_input_shape"][1]
        
    elif entry["class_name"] == "Dense":
        layer_conf = entry["config"]
        layers[layer_idx] = {"units": layer_conf["units"],
                     "units_prev": units_prev,
                     "activation": layer_conf["activation"]}
        layers["num_layers"] += 1
        units_prev = layer_conf["units"]
        layer_idx += 1
        
        pass
    else:
        raise Exception("Unknown Type of Layer")
        pass
    
    pass

#%%

DATA_WIDTH = 16
DATA_Q = 8
FP_ONE = 2**DATA_Q

inst_port_maps = [""]*layers["num_layers"]


for i in range(layers["num_layers"]):
    j = 2*i
    
    w = weights[j] * FP_ONE
    w = np.int64(w)
    w = np.transpose(w)
    
    b = weights[j+1] * FP_ONE
    b = np.int64(b)
    
    layer = layers[i]
    
    inst_port_maps[i]=(
        f"U_{i} : c_004_layer_01" + "\n"+
        "generic map (" + "\n"+
        f"g_layer_length_cur => {layer['units']}," + "\n"+
        f"g_layer_length_prev => {layer['units_prev']}," + "\n" )
    
    if len(b) == 1:
        txt_bias = f"g_layer_bias => ( 0 => {b[0]} )," + "\n"
    else:
        
        pass
    
    
    





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



