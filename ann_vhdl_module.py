# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:53:03 2025

@author: Admin
"""
#%% IMPORTS
# import os
import datetime

#%%

class ann_vhdl_machine:
    _version = 1
    
    def __init__(self, name_entity):
        self._entity = {}
        self._entity["name"] = name_entity
        self._entity["port"] = {}
        self._entity["library"] = {}
        
        # self._dict_arch = {name_arch:{}}
        # self._arch_current = name_arch
        # self._dict_signals = {}
        pass
    
    def port_add(self, name, direction, data_type):
        ports = self._entity["port"]
        
        if name in ports:
            raise Exception(f"Port name [{name}] has already been added!")
            pass
        
        ports[name] = {"direction": direction,
                       "data_type": data_type}
        pass
    
    def entity_library_set(self, lib_name):
        libs = self._entity["library"]
        
        if lib_name in libs:
            raise Exception(f"Library name [{lib_name}] has already been set!"); pass
        libs[lib_name] = []
        pass
    
    def entity_library_use(self, lib_name, use_name):
        libs = self._entity["library"]
        
        if lib_name not in libs:
            raise Exception(f"Library name [{lib_name}] has not been set yet!"); pass
        
        sub_lib = libs[lib_name]
        if use_name in sub_lib:
            raise Exception(f"Use Case of [{lib_name}.{use_name}] has already been set!"); pass
        sub_lib.append(use_name)
        pass
    
    
    
    
    
    # def signal_declare(self, name, data_type):
    #     if name in self._dict_signals:
    #         raise Exception(f"Signal name [{name}] already exists!")
    #         pass
        
    #     self._dict_signals[name] = data_type
    #     pass
    
    def to_string(self):
        
        temp_list = []
        
        temp_list.extend( self._to_string_entity_head() )
        temp_list.extend( self._to_string_entity_libs() )
        
        
        return temp_list
    
    def _to_string_entity_head(self):
        entity = self._entity
        return [
            f"-- VHDL Entity {entity['name']}",
            f"-- {datetime.datetime.now()}",
            f"-- VERSION: {ann_vhdl_machine._version}",
            "--"
        ]
    
    def _to_string_entity_libs(self):
        entity = self._entity
        libs = entity["library"]
        temp_list = []
        
        for lib_name in libs:
            temp_list.append(f"LIBRARY {lib_name};")
            
            for use_name in libs[lib_name]:
                temp_list.append(f"USE {lib_name}.{use_name};")
                pass
            pass
        return temp_list
    
    def _to_string_entity_main(self):
        entity = self._entity
        temp_list = []
        
        temp_list.append("entity {entity[name]} is")
        
        temp_list.extend( self._to_string_entity_port() )
        
        temp_list.append("end {entity[name]};")
        pass
    
    def _to_string_entity_port(self):
        entity = self._entity
        ports = entity["port"]
        temp_list = []
        
        pass
    pass


#%%
if __name__ == "__main__":
    
    myTest = ann_vhdl_machine("tb_004_layer")
    
    myTest.entity_library_set("ieee")
    myTest.entity_library_use("ieee", "std_logic_1164.all")
    myTest.entity_library_use("ieee", "numeric_std.all")
    
    myTest.port_add("clk",          "in", "std_logic")
    myTest.port_add("reset",        "in", "std_logic")
    
    myTest.port_add("src_TX_0",     "in",  "std_logic")
    myTest.port_add("ack_RX_0",     "out", "std_logic")
    
    myTest.port_add("dst_RX_n",      "in",  "std_logic")
    myTest.port_add("ready_to_TX_n", "out", "std_logic")
    
    myTest.port_add("layer_in_0",   "in",  "t_array_data_stdlv (0 to 1)")
    myTest.port_add("layer_out_n",  "out", "t_array_data_stdlv (0 to 1)")
    
    a = myTest.to_string()
    pass
