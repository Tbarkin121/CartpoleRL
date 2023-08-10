# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:57:32 2023

@author: Plutonium
"""

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from cartpole import CartPole
from cartpole_fourplayer import CartPole_4P
from buffer import Buffer
import time

from torchviz import make_dot
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

num_envs = 5000
horizon = 25
gamma = 0.99

num_epochs = 10000
num_q_holds = 1
# minibatch_steps = 1
# minibatch_size = 1000

take_n_actions = 10
entropy_coff_inital = 0.0
clip_range = 0.2
            
torch.set_default_device('cuda')

#%%
# Define model
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define actor and critic networks
        
        n_features = 20
        n_actions = 4
        layer1_count = 256
        layer2_count = 128
        layer3_count = 64

        
        self.shared1 = nn.Sequential(
                                    nn.Linear(n_features, layer1_count),
                                    nn.ELU()
                                    )
        
        self.shared2 = nn.Sequential(
                                    nn.Linear(layer1_count+n_features, layer2_count),
                                    nn.ELU()
                                    )
        
        self.policy1 = nn.Sequential(
                                    nn.Linear(layer2_count+n_features, layer3_count),
                                    nn.ELU()
                                    )
        self.policy2 = nn.Sequential(
                                    nn.Linear(layer3_count+n_features, n_actions),
                                    nn.Tanh(),
                                    )
        
        self.value1 = nn.Sequential(
                                    nn.Linear(layer2_count+n_features, layer3_count),
                                    nn.ELU()
                                    )
        self.value2 = nn.Sequential(
                                    nn.Linear(layer3_count+n_features, 1),
                                    )
        
        # self.policy = nn.Sequential(
        #                             nn.Linear(n_features, layer1_count),
        #                             nn.ELU(),
        #                             nn.Linear(layer1_count, layer2_count),
        #                             nn.ELU(),
        #                             nn.Linear(layer2_count, n_actions),
        #                             nn.Tanh(),
        #                             )
        # self.value = nn.Sequential(
        #                             nn.Linear(n_features, layer1_count),
        #                             nn.ELU(),
        #                             nn.Linear(layer1_count, layer2_count),
        #                             nn.ELU(),
        #                             nn.Linear(layer2_count, 1),
        #                             )

    def forward(self, x):
        s1 = torch.cat((self.shared1(x), x), dim=-1)
        s2 = torch.cat((self.shared2(s1), x), dim=-1)
        v1 = torch.cat((self.value1(s2) , x), dim=-1)
        v2 = self.value2(v1) 
        p1 = torch.cat((self.policy1(s2), x), dim=-1)
        p2 = self.policy2(p1)        
        # p = self.policy(x)
        # v = self.value(x)
        return v2, p2
    

    
#%%
# a = torch.rand([3,5])
Agent = Policy()
agent_optim = optim.Adam(Agent.parameters(), lr=3e-4)

# # actor_optim = optim.Adam( [ { 'params': PolicyNet1.actor_top.parameters() },
# #                             { 'params': PolicyNet1.actor_mean.parameters() },
# #                             { 'params': PolicyNet1.actor_std.parameters()} ], 
# #                             lr=1e-3)
          


#%%
                          
mse_loss = torch.nn.MSELoss()
# env = CartPole(num_envs = num_envs, buf_horizon=horizon, gamma=gamma, rand_reset=True)
env = CartPole_4P(num_envs = num_envs, buf_horizon=horizon, gamma=gamma, rand_reset=True)
replay_buffer = Buffer(env.buffer_hor, num_envs, env.num_actions, env.num_states*env.num_players, gamma)
env.register_buffer(replay_buffer)


log_probs = torch.zeros((num_envs, horizon))
log_probs_old = torch.zeros((num_envs, horizon)).detach()
env.render_init()


#%%

# for epoch in range(1):
for epoch in range(num_epochs):

    # PolicyNet2.load_state_dict(PolicyNet1.state_dict())
    # Critic2.load_state_dict(Critic.state_dict())
    actor_loss_list = []
    critic_loss_list = []
    

    for _ in range(1):
    # for _ in range(minibatch_steps):
        # s1, a1, r1, s2, d2, log_probs_old, returns = env.buffer.get_SARS_minibatch(minibatch_size) 
        s1, a1, r1, s2, d2, log_probs_old, returns = replay_buffer.get_SARS()

        
        [vals_s1, probs_s1] = Agent(s1)
        
        
        # action_pd_s1 = torch.distributions.Normal(probs_s1[:,:,0], probs_s1[:,:,1])
        action_pd_s1 = torch.distributions.Normal(probs_s1, 0.1*torch.ones_like(probs_s1.detach()))
        

        # td_error = returns - vals_s1.squeeze(-1)
        advantage = returns - vals_s1.squeeze(-1)
        

        # normalize advantage... (Doesn't Seem to work)
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        
        log_probs = action_pd_s1.log_prob(a1)
        ratio = torch.exp(log_probs - log_probs_old)
        

        entropy_coff = entropy_coff_inital * (1-(epoch/num_epochs))
        # entropy_loss = -action_pd_s1.entropy().mean() * entropy_coff
        entropy_loss = -entropy_coff*torch.mean(-log_probs)

        policy_loss_1 = advantage.view(env.num_envs, env.buffer_hor, 1).detach() * ratio
        policy_loss_2 = advantage.view(env.num_envs, env.buffer_hor, 1).detach() * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean() 
        
        value_loss = mse_loss(vals_s1.squeeze(-1), returns)
        
        total_loss = policy_loss + value_loss*0.2 + entropy_loss
        
        agent_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(Agent.parameters(), 0.5) #Max Grad Norm
        agent_optim.step()

        
        actor_loss_list.append(policy_loss.detach().cpu().numpy())
        critic_loss_list.append(value_loss.detach().cpu().numpy())
        
        

        
        for name, param in Agent.named_parameters():
            if( torch.any(torch.isnan(param)) ):
                print(name)
                print(param)
                BREAKPOINTBULLSHIT

            
        
        # print('---')
        # print('s1')
        # print(s1)
        # print('a1')
        # print(a1)
        # print('r1')
        # print(r1)
        # print('s2')
        # print(s2)
        # print('d2')
        # print(d2)
        # print('advantage')
        # print(advantage)
        # print('RETURNS')
        # print(returns)
        
        
    
    with torch.no_grad():
        env.render_init()
        for _ in range(take_n_actions):
            env.render(0)
            
            s1, a1, r1, s2, d2, log_probs_old, returns = replay_buffer.get_SARS()
            [vals, probs] = Agent(s2)
            newest_probs = probs[:,0,:]
            action_pd = torch.distributions.Normal(newest_probs, 0.1*torch.ones_like(newest_probs))
            next_actions = action_pd.sample()
            log_probs_sample = action_pd.log_prob(next_actions)
            env.step(next_actions, log_probs_sample, Agent)
        
            

    print(epoch)
    print('Policy Loss Avg: {}. Value Loss Avg: {}. Avg Returns: {}'.format(np.array(actor_loss_list).mean(), np.array(critic_loss_list).mean(), returns.mean()))
    
    with torch.no_grad():
        # Useful extra info
        approx_kl1 = ((torch.exp(ratio) - 1) - ratio).mean() #Stable Baselines 3
        approx_kl2 = (log_probs_old - log_probs).mean()    #Open AI Spinup
        # print('kl approx : {} : {} : {}'.formaWDt(approx_kl1, approx_kl2, ratio.mean()))
    
    # print(Qvals.reshape([siz,siz]))
    
torch.save(Agent.state_dict(), "model_name.pth")


#%%
import time
num_envs=2
test_env = CartPole_4P(num_envs = num_envs, buf_horizon=10, gamma=gamma, rand_reset=True)
test_replay_buffer = Buffer(test_env.buffer_hor, num_envs, test_env.num_actions, test_env.num_states*test_env.num_players, gamma)
test_env.register_buffer(test_replay_buffer)


test_env.render_init()
Agent.load_state_dict(torch.load("model_name.pth"))
#%%
env_ids = torch.arange(num_envs)
test_env.reset_idx(env_ids)

test_env.state[0,:] = 0
test_env.state[0,2] = 10*torch.pi/180.0
#%%

view_len = 1000
env_view_id = 0
for i in range(view_len):
    start_time = time.perf_counter()
    test_env.render(env_view_id)
    s1, a1, r1, s2, d2, log_probs_old, returns = test_replay_buffer.get_SARS() 
    
    
    # probs = Actor(s2)
    [vals, probs] = Agent(s2)
    newest_probs = probs[:,0,:]
    action_pd = torch.distributions.Normal(newest_probs, 0.1*torch.ones_like(newest_probs))
    
    a = test_env.joy.get_axis()
    next_actions = probs[:,0,:] + a[0]
    
    log_probs = action_pd.log_prob(next_actions)
    
    test_env.step(next_actions, log_probs, Agent)

    time_diff = time.perf_counter() - start_time
    print('S{} : {}s'.format(i, time_diff))
    # print(next_actions[0,0])
    # if(d2[0,0]):
    #     print('RESET')


#%%

input_names = ["state_scaled"]
output_names = ["value"] + ["mean"]
dummy_input = torch.randn([1, 5])
torch.onnx.export(Agent, dummy_input, "cartpole_256_128_64.onnx", verbose=True, input_names=input_names, output_names=output_names)


#%%

test_input = torch.tensor([0.86, 0.486, -0.476, 0.0579, 0.400]).view(1,-1)
[test_v, test_p] = Agent(test_input)

print('{} : {}'.format(test_v, test_p))

#%%
# STM32 Inference

import serial
import struct

port = "COM9"
baudrate = 256000 
timeout = 1
view_len = 1000
env_view_id = 0

time_accumulator = []
with serial.Serial(port, baudrate, timeout=timeout) as ser:
    for i in range(view_len):
        start_time = time.perf_counter()
        test_env.render(env_view_id)
        s1, a1, r1, s2, d2, log_probs_old, returns = replay_buffer.get_SARS() 
        
       
        packet = bytearray()
        for d in s2[0,0,:]:
            for b in struct.pack("f", d):
                packet.append(b)
                
        ser.write(packet)
        ser.flush()
       
        x = ser.read(8)
        val = struct.unpack('f', x[0:4])[0]
        act = struct.unpack('f', x[4:8])[0]
        
        
        a = test_env.joy.get_axis()
        next_actions = torch.ones([num_envs,1])*act + a[0]
        
        log_probs = action_pd.log_prob(next_actions)
        
        test_env.step(next_actions, log_probs, Agent)
        
        time_diff = time.perf_counter() - start_time
        time_accumulator.append(time_diff)
        print('S{} : {}s'.format(i, time_diff))

time_accum = torch.tensor(time_accumulator)
mean_time = time_accum.mean()
print("Average Loop Time : {}s. ".format(mean_time))
print("Average Loop Freq : {}Hz. ".format(1/mean_time))

#%%
import matplotlib.pyplot as plt

data = Agent.shared1[0].weight.sort(dim=1)[0].cpu().detach().numpy()
# plt.plot(data)
# plt.show()
# for name, param in Agent.named_parameters():
#     if( torch.any(torch.isnan(param)) ):
#         print(name)
#         print(param)
#         BREAKPOINTBULLSHIT
        
        