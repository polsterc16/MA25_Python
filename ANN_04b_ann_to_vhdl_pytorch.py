# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:25:11 2025

@author: Admin
"""
#%% IMPORTS
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as data

# import onnx


#%%
path_out_dir = os.path.join("OUTPUT", "04_ann_to_vhdl")
# Check whether the specified path exists or not
isExist = os.path.exists(path_out_dir)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(path_out_dir)
    print("The new directory is created!"+"\n"+f"{path_out_dir}")

del isExist



#%% Load Model
# onnx_model = onnx.load("03c_ann_pytorch_model.onnx")


""" INFO: We expect to only use loaded models via torch.jit.load of a TorchScript model! 
    Therefore, we will rely on the 'original_name' attribute of the model children,
    to decide if they represent a valid network for this script."""

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
model = torch.jit.load('03c_ann_pytorch_model.pt')
model.eval()


assert hasattr(model, 'original_name')
model_name = model.original_name


# fetch children of the model
chil_model = [ch for ch in model.children()]


# CHECK: We expect this to only contain 1 "Sequential" module als child (see 03c_ann_pytorch)
assert len(chil_model)==1 # must be only one element: a Sequential layer stack
mod = chil_model[0]


# CHECK: We expect it to be of the type "Sequential" (see 03c_ann_pytorch)
assert hasattr(mod, 'original_name')
assert mod.original_name == "Sequential"

chil_seq = [ch for ch in mod.children()]
ann_layers = chil_seq

#%%
list_layers = ["Linear",]
# list_activation = ["ReLU", "Hardsigmoid"]

# dict_map_pytorch_layer = {}


dict_map_activation = {
    "ReLU": "RELU",
    "Hardsigmoid": "HARD_SIGMOID",
}


layers = {"num_layers":0}

for i,entry in enumerate(ann_layers):
    n = entry.original_name
    print("-- Layer", i, f"[{n}]")
    
    if n in list_layers:
        # if is DENSE / Fully Connected Layer
        if n == "Linear":
            # layer_conf = entry["config"]
            layers[layers["num_layers"]] = {
                "activation": "IDENTITY",
                "pytorchIdx": i,
            }
            # dict_map_pytorch_layer[layers["num_layers"]] = i
            
            layers["num_layers"] += 1
            pass
        # else: if is conv layer, todo:future
    elif n in dict_map_activation:
        # if is Activation function (of prev layer)
        
        if n == "ReLU":
            layers[layers["num_layers"]-1]["activation"] = dict_map_activation[n]
            
        elif n == "Hardsigmoid":
            layers[layers["num_layers"]-1]["activation"] = dict_map_activation[n]
        
    else:
        raise Exception("Unknown Type of Layer")
        pass
    
    pass

###############################################################################

state_dict = model.state_dict()
postfix_weight = ".{}.weight"
postfix_bias   = ".{}.bias"

# we must get the name of our dict entries
list_keys = [k for k in state_dict]
# so we fetch the first element and split with "." as separator
prefix = list_keys[0].split(".")[0]

for idx in range(layers["num_layers"]):
    entry = layers[idx]
    print(idx,entry)
    
    # the dict keys are a composite of prefix and postfix (formated with pytorchIdx)
    pytorchIdx = entry["pytorchIdx"]
    key_weights = prefix + postfix_weight.format(pytorchIdx)
    key_biases  = prefix + postfix_bias.format(pytorchIdx)
    
    entry["weights"] = state_dict[key_weights].numpy()
    entry["biases"] = state_dict[key_biases].numpy()
    # entry["units"]
    
    entry["units"], entry["units_prev"] = entry["weights"].shape
    
    
    



#%%

raise Exception()

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
    
    weights = layer["weights"]
    w = weights * FP_ONE
    w = np.int64(w)
    # raise Exception()
    # w = np.transpose(w)
    
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



