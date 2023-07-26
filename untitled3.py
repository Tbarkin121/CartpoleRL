# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:58:41 2023

@author: MiloPC
"""

import torch
import time

import os


with torch.no_grad():
    
    torch.set_default_device('cuda')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    m = 10000
    n = 10
    n_step = 3
    buffer_horizon = 100
    gamma = 0.9
    
    states = torch.rand([buffer_horizon,m,n])
    values = torch.rand([m,1,buffer_horizon])
    values_mat = values.squeeze()
    rewards = torch.rand([buffer_horizon,m,1])
    GAMMA = gamma**torch.linspace(0,5,n_step+1).reshape([1,1,-1])
    
    GAMMA_mat = torch.zeros(buffer_horizon,buffer_horizon-n_step)
    
    for i in range(buffer_horizon-n_step):
        GAMMA_mat[i:n_step+1+i,i] = GAMMA.reshape([-1])



    
    a1 = torch.zeros((m,1,buffer_horizon-n_step))
    a2 = torch.zeros((m,buffer_horizon-n_step))
    
    for _ in range(3):
        now2 = time.perf_counter()
        a2 = torch.matmul(values_mat,GAMMA_mat)
        dt2 = time.perf_counter() - now2
        
        print(dt2)
        
        
    for _ in range(3):
        now1 = time.perf_counter()
        a1 = torch.nn.functional.conv1d(values, GAMMA)
        dt1 = time.perf_counter() - now1
        
        print(dt1)
