# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:19:59 2023

@author: Plutonium
"""

import torch

num_env = 2
horizon = 10
num_states = 2

test_buff = torch.zeros((horizon, num_env, num_states))

print(test_buff)





for _ in range(horrizon):
    sample_state = torch.rand((num_env, num_states))
    test_buff[0, :, :] = sample_state
    test_buff = test_buff.roll(-1, 1)
    print(test_buff)
    
    
    
    
    
#%%

print(test_buff[0,1])