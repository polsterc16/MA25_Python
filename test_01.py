# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:17:40 2025

@author: Admin
"""
#%% IMPORTS
import os

import numpy as np
import matplotlib.pyplot as plt
# import pickle
# import json
#%%
if "np" in locals(): print("--", "numpy version:\t", np.__version__)
#%%
path_OUTPUT = "OUTPUT/test_01"

if not os.path.isdir(path_OUTPUT):
    os.makedirs(path_OUTPUT) 

#%%
fname_file = "test.vhd"
path_file = os.path.join(path_OUTPUT, fname_file)

lines = (
    "LIBRARY ieee;",
    "USE ieee.std_logic_1164.all;",
    "USE ieee.numeric_std.all;",
)


with open(path_file, "w") as f:
    for line in lines:
        f.write(line+"\n")
        
    pass




#%%

