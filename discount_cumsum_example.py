# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:04:22 2023

@author: Plutonium
"""

import torch
import torchaudio.functional as ta_fun
import matplotlib.pyplot as plt
import time 


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    out = ta_fun.lfilter(x, torch.tensor([1.0, -discount]), torch.tensor([1.0, 0.0]), clamp=False)
    return out





num_envs = 100000
horizon = 10
gamma = 0.9

rewards = torch.ones(num_envs, horizon)
vals = torch.rand(num_envs, horizon)


# print(rewards)
# print(vals)

start_time = time.perf_counter()
delta = rewards[:-1] + gamma * vals[1:] - vals[:-1]

# print(delta)

advantage = discount_cumsum(delta, gamma)

end_time = time.perf_counter()

print('Advantage Calc Time: {}. FPS: {}'.format(end_time-start_time, num_envs/(end_time-start_time)))


#%%



num_envs = 3
horizon = 10
data = torch.ones(num_envs, horizon)
data[1,:] *= 2
data[2,:] *= 3

gamma = 0.9
out = discount_cumsum(data, gamma)

print('DATA')
print(data)
print('OUT')
print(out)

plt.plot(data.permute([1,0]).detach().cpu().numpy())
plt.plot(out.permute([1,0]).detach().cpu().numpy())
plt.ylabel('some numbers')
plt.grid(True)
plt.show()
plt.legend(['1', '2'])
