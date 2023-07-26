# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:53:34 2023

@author: Plutonium
"""


import torch

gamma = 0.9 

num_envs = 2
horizon = 5
gamma_mask = torch.ones((horizon+1)) * gamma
gamma_mask = torch.cumprod(gamma_mask, dim=0)/gamma
print(gamma_mask)

rewards_2_go = torch.zeros((num_envs, horizon))
dones =  torch.ones((num_envs, horizon))==0
dones_old =  torch.ones((num_envs, horizon))==0


#%%
for _ in range(1):
    dones = dones.roll(1,1)
    dones[:,0] = torch.rand((num_envs,))>0.90
    
    dones_tmp = dones.clone()
    dones_tmp[:,0] = False
    
    dones_mask = torch.where(dones_tmp, 0, 1)
    dones_mask = torch.cumprod(dones_mask, dim=1)
    
    print('dones')
    print(dones)
    print(dones_mask)
    
    
    
    
    
    # reward = torch.rand((num_envs,1))
    reward = torch.ones((num_envs,1))
    reward = torch.where(dones[:,0].view(-1, 1), -1, reward)
    print('reward')
    print(reward)

    
    rewards_2_go = rewards_2_go.roll(1,1)
    # advantages[:,0] = reward.view(1,-1)
    rewards_2_go[:,0] = 0
    rewards_2_go += dones_mask*gamma_mask[0:horizon]*reward
    
    print('rewards_2_go')
    print(rewards_2_go)
    
    dones_old = dones.clone()
    
    
    dones_mask2 = torch.where(dones, 0, 1)
    dones_mask2 = torch.cumprod(dones_mask2, dim=1)
    value_gamma_scaler = dones_mask2*gamma_mask[1:]
    

    print('value_gamma_scaler')
    print(value_gamma_scaler)