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

del isExist
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

del units_prev, layer_idx, n, entry

#%%


#%%

DATA_WIDTH = 16
DATA_Q = 8
FP_ONE = 2**DATA_Q

inst_port_maps = [""]*layers["num_layers"]

list_signals = []

list_port_inout = []


for i in range(layers["num_layers"]):

    layer = layers[i]
    
    w = weights[2*i] * FP_ONE
    w = np.int64(w)
    w = np.transpose(w)
    
    b = weights[2*i + 1] * FP_ONE
    b = np.int64(b)
    
    
    txt_component =  f"U_{i} : c_004_layer_01" + "\n{};\n"
    
    
    txt_generic = "generic map (\n{}\n)"
    txt_port = "port map (\n{}\n)"
    
    
    
    
    #-------
    #------- GENERIC MAP
    #-------    
    
    list_generics = []
    
    list_generics.append("  "+f"g_layer_length_cur => {layer['units']}")
    #-------
    list_generics.append("  "+f"g_layer_length_prev => {layer['units_prev']}")
    #-------
    
    # Bias: is 1D array.
    # We take care to correctly format, if the array only contains 1 element
    txt_b1 = "( {} )"
    if len(b) == 1:
        txt_b1 = "( 0 => {} )"
    
    b1 = [str(x) for x in b]
    b1 = txt_b1.format(", ".join(b1))
    txt_bias = "  "+f"g_layer_bias => {b1}"
    list_generics.append(txt_bias)
    #-------
    
    # Weights: is a 2d array, where the first index is the current unit
    # and the second index is the previous unit.
    # We take care to correctly format, if an array only contains 1 element
    
    txt_w1 = "( {} )"
    if len(w) == 1:
        txt_w1 = "( 0 => {} )"
        pass
    
    list_w2 = []
    for elem in w:
        txt_w2 = "({})"
        if len(elem) == 1:
            txt_w2 = "( 0 => {} )"
            pass
        
        w2 = [str(x) for x in elem]
        w2 = txt_w2.format(", ".join(w2))
        list_w2.append(w2)
        pass
    w1 = txt_w1.format(", ".join(list_w2))
    
    txt_weights = "  "+f"g_layer_weights => {w1}"
    list_generics.append(txt_weights)
    #-------
    
    
    list_generics.append("  "+f"g_act_func => AF_{layer['activation'].upper()}")
    #-------
    
    
    
    txt_generic = txt_generic.format(",\n".join(list_generics))
    del list_generics, txt_weights, list_w2, txt_w1, txt_w2, w, w1, w2
    del txt_bias, txt_b1, b1, b, elem
    
    # print("\n--",i)
    # print(txt_generic)
    
    
    
    
    
    
    #-------
    #------- PORT MAP
    #------- 
    
    # https://asciiflow.com/
    #     ┌───────────────────────┐    
    #     │                       │    
    # ───>│ CLK                   │    
    #     │                       │    
    # ───>│ RESET                 │    
    #     │                       │    
    #     │                       │    
    # <───│ ACK_RX         DST_RX │<───
    #     │                       │    
    # ───>│ SRC_TX           R2TX │───>
    #     │                       │    
    #     │                       │    
    # ───>│ Layer_in    Layer_out │───>
    #     │                       │    
    #     └───────────────────────┘    
    
    signal_decl_DST_RX = ""
    signal_decl_R2TX = ""
    signal_decl_Layer = ""
    if i == 0:
        signal_ACK_RX    = "ack_RX"
        signal_SRC_TX    = "src_TX"
        signal_Layer_in  = "layer_in"
        
        signal_DST_RX    = f"DST_RX_{i}"
        signal_R2TX      = f"R2TX_{i}"
        signal_Layer_out = f"layer_{i}"
        
        signal_decl_DST_RX = f"signal {signal_DST_RX} : std_logic;"
        signal_decl_R2TX   = f"signal {signal_R2TX} : std_logic;"
        signal_decl_Layer  = f"signal {signal_Layer_out} : t_array_data_stdlv(0 to {layer['units'] - 1});"
        list_signals.append(signal_decl_DST_RX)
        list_signals.append(signal_decl_R2TX)
        list_signals.append(signal_decl_Layer)
        
        list_port_inout.append(f"layer_in : in t_array_data_stdlv (0 to {layer['units_prev'] - 1})")
        pass
    
    elif i == (layers["num_layers"]-1):
        signal_ACK_RX    = f"DST_RX_{i-1}"
        signal_SRC_TX    = f"R2TX_{i-1}"
        signal_Layer_in  = f"layer_{i-1}"
        
        signal_DST_RX    = "dst_RX"
        signal_R2TX      = "ready_to_TX"
        signal_Layer_out = "layer_out"
        
        list_port_inout.append(f"layer_out : out t_array_data_stdlv (0 to {layer['units'] - 1})")
        pass
    
    else:
        signal_ACK_RX    = f"DST_RX_{i-1}"
        signal_SRC_TX    = f"R2TX_{i-1}"
        signal_Layer_in  = f"layer_{i-1}"
        
        signal_DST_RX    = f"DST_RX_{i}"
        signal_R2TX      = f"R2TX_{i}"
        signal_Layer_out = f"layer_{i}"
        
        signal_decl_DST_RX = f"signal {signal_DST_RX} : std_logic;"
        signal_decl_R2TX   = f"signal {signal_R2TX} : std_logic;"
        signal_decl_Layer  = f"signal {signal_Layer_out} : t_array_data_stdlv(0 to {layer['units'] - 1});"
        list_signals.append(signal_decl_DST_RX)
        list_signals.append(signal_decl_R2TX)
        list_signals.append(signal_decl_Layer)
        pass
    
    
    
        
    list_ports = []
    
    list_ports.append("  "+f"clk => clk")
    list_ports.append("  "+f"reset => reset")
    #-------
    list_ports.append("  "+f"ack_RX => {signal_ACK_RX}")
    list_ports.append("  "+f"src_TX => {signal_SRC_TX}")
    list_ports.append("  "+f"layer_in => {signal_Layer_in}")
    #-------
    list_ports.append("  "+f"dst_RX => {signal_DST_RX}")
    list_ports.append("  "+f"ready_to_TX => {signal_R2TX}")
    list_ports.append("  "+f"layer_out => {signal_Layer_out}")
    
    # print(list_ports)
    txt_port = txt_port.format(",\n".join(list_ports))
    
    # print(txt_port)
    del signal_ACK_RX, signal_SRC_TX, signal_Layer_in
    del signal_DST_RX, signal_R2TX, signal_Layer_out
    del signal_decl_DST_RX, signal_decl_R2TX, signal_decl_Layer
    del list_ports
    
    
    
    
    #-------
    #------- Complete Component
    #------- 
    
    txt_component =  txt_component.format("\n".join([txt_generic, txt_port]))
    # print(txt_component)
    inst_port_maps[i] = txt_component
    
    del txt_component, txt_generic, txt_port
    
    pass
    
#-------
#------- Signals
#------- 
txt_signals = "\n".join(list_signals)
del list_signals


#-------
#------- Port Layer in out
#------- 
txt_port_layer_inout = ";\n".join(list_port_inout)
del list_port_inout



inst_port_maps = "\n".join(inst_port_maps)
print(inst_port_maps)



#%%





#%%

path_in_dir = os.path.join("INPUT", "04_ann_to_vhdl")
 
if not os.path.exists(path_in_dir):
    raise Exception(f"Path does not exist: \n{path_in_dir}")
    pass

file_name = "Combined v2.vhd"
fPathIn = os.path.join(path_in_dir, file_name)
# D:\DCD_workspace\python\MA25_Python\INPUT\04_ann_to_vhdl\Combined v2.vhd

#%%

with open(fPathIn) as f:
    fileStr = f.read()
    pass

EntityName = "c_x_ANN_01"
fileNameOut = f"{EntityName}.vhd"
fPathOut = os.path.join(path_out_dir, fileNameOut)

fileStr = fileStr.replace("{$NAME_ENTITY}", EntityName)


tnow = datetime.datetime.now()
fileStr = fileStr.replace("{$DATE_TIME}", str(tnow))



fileStr = fileStr.replace("{$SIGNAL_DECLARATION}", txt_signals)



fileStr = fileStr.replace("{$INSTANCE_PORT_MAPPINGS}", inst_port_maps)





fileStr = fileStr.replace("{$PORT_LAYER_IN_OUT}", txt_port_layer_inout)




with open(fPathOut, mode="w") as f:
    f.write(fileStr)

#%%



