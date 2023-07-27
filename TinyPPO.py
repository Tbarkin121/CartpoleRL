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
import time

from torchviz import make_dot
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

num_envs = 2
horizon = 5
gamma = 0.9

num_epochs = 1000
num_q_holds = 1

minibatch_steps = 1
minibatch_size = 1000
take_n_actions = 1
entropy_coff_inital = 1.0
clip_range = 0.2
            
torch.set_default_device('cuda')

#%%
# Define model
class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define actor and critic networks
        
        n_features = 5
        n_actions = 1
        
        critic_layers = [
                        nn.Linear(n_features, 128),
                        nn.LeakyReLU(),
                        nn.Linear(128, 1),  # estimate V(s)
                    ]
                    
        actor_top_layers = [
                        nn.Linear(n_features, 128),
                        nn.LeakyReLU(),
                    ]
        
        actor_mean_layers = [
                        nn.Linear(128, n_actions),
                        nn.Tanh(),
                    ]
        
        # actor_std_layers = [
        #                 nn.Linear(128, n_actions),
        #                 # nn.Softplus(),
        #                 nn.Sigmoid(),
        #             ]
        
        
        self.critic = nn.Sequential(*critic_layers)
        self.actor_top = nn.Sequential(*actor_top_layers)
        self.actor_mean = nn.Sequential(*actor_mean_layers)
        # self.actor_std = nn.Sequential(*actor_std_layers)


    def forward(self, x):
        state_values = self.critic(x)  # shape: [n_envs,]
        a0 = self.actor_top(x)  # shape: [n_envs, n_actions]
        mean = self.actor_mean(a0)
        # std = self.actor_std(a0) + 0.01
        # action_distribution_vec = torch.cat((mean, std), dim=2)
        
        return (state_values, mean)
    
class ValueNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define actor and critic networks
        
        n_features = 5
        n_actions = 1
        
        layers = [nn.Linear(n_features, 128),
                  nn.LeakyReLU(),
                  nn.Linear(128, 128),  
                  nn.LeakyReLU(),
                  nn.Linear(128, 64),  
                  nn.LeakyReLU(),
                  nn.Linear(64, n_actions),  
                 ]
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        values = self.layers(x)  # shape: [n_envs,]
        return (values)
    
class PolicyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # define actor and critic networks
        
        n_features = 5
        
        layers = [nn.Linear(n_features, 128),
                  nn.LeakyReLU(),
                  nn.Linear(128, 128),  
                  nn.LeakyReLU(),
                  nn.Linear(128, 64),  
                  nn.LeakyReLU(),
                  nn.Linear(64, 1), 
                  nn.Tanh(),
                 ]
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        values = self.layers(x)  # shape: [n_envs,]
        return (values)

    
#%%
# a = torch.rand([3,5])
# PolicyNet1 = Policy()

# # critic_optim = optim.Adam(PolicyNet1.critic.parameters(), lr=1e-3)

# # actor_optim = optim.Adam( [ { 'params': PolicyNet1.actor_top.parameters() },
# #                             { 'params': PolicyNet1.actor_mean.parameters() },
# #                             { 'params': PolicyNet1.actor_std.parameters()} ], 
# #                             lr=1e-3)


# # acer_optim = optim.Adam( [  { 'params': PolicyNet1.critic.parameters() },
# #                             { 'params': PolicyNet1.actor_top.parameters() },
# #                             { 'params': PolicyNet1.actor_mean.parameters() },
# #                             { 'params': PolicyNet1.actor_std.parameters()} ], 
# #                             lr=1e-3)
                             
                             
# acer_optim = optim.Adam(PolicyNet1.parameters(), lr=1e-3)

# PolicyNet2 = Policy()
# PolicyNet2.eval()

mse_loss = torch.nn.MSELoss()
env = CartPole(num_envs = num_envs, buf_horizon=horizon, gamma=gamma, rand_reset=False)
log_probs = torch.zeros((num_envs, horizon))
log_probs_old = torch.zeros((num_envs, horizon)).detach()
env.render_init()

#%%

Critic = ValueNet()
Critic2 = ValueNet()
Actor  = PolicyNet()

Critic2.eval()

critic_optim = optim.Adam(Critic.parameters(), lr=1e-3)
actor_optim = optim.Adam(Actor.parameters(), lr=1e-3)
#%%

# for epoch in range(1):
for epoch in range(num_epochs):

    # PolicyNet2.load_state_dict(PolicyNet1.state_dict())
    # Critic2.load_state_dict(Critic.state_dict())
    actor_loss_list = []
    critic_loss_list = []
    
    # for i in range(1):
    for i in range(num_q_holds):
        # s1, a1, r1, s2, d2 = env.buffer.get_SARS() 
        
        for _ in range(1):
        # for _ in range(minibatch_steps):
            # s1, a1, r1, s2, d2 = env.buffer.get_SARS_minibatch(minibatch_size) 
            s1, a1, r1, s2, d2, log_probs_old, returns = env.buffer.get_SARS()


            vals_s1 = Critic(s1)
            vals_s2 = Critic(s2) # We only should need these for the current state of the buffer
            probs_s1 = Actor(s1)
            
            
            # [vals_s1, probs_s1] = PolicyNet1(s1)
            # [vals_s2, probs_s2] = PolicyNet2(s2)
            
            
            # action_pd_s1 = torch.distributions.Normal(probs_s1[:,:,0], probs_s1[:,:,1])
            action_pd_s1 = torch.distributions.Normal(probs_s1[:,:,0], 0.05*torch.ones_like(probs_s1[:,:,0].detach()))
            
            
            # returns = env.buffer.value_gamma_scaler*vals_s2[:,0] + r1
            # print('Returns 1 : ')
            # print(returns)
            
            # td_error = returns - vals_s1.squeeze(-1)
            advantage = returns - vals_s1.squeeze(-1)
            

            # normalize advantage... (Doesn't Seem to work)
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            
            log_probs = action_pd_s1.log_prob(a1)
            ratio = torch.exp(log_probs - log_probs_old)
            

            entropy_coff = entropy_coff_inital * (1-(epoch/num_epochs))
            # entropy_loss = -action_pd_s1.entropy().mean() * entropy_coff
            entropy_loss = -entropy_coff*torch.mean(-log_probs)
    
            policy_loss_1 = advantage.detach() * ratio
            policy_loss_2 = advantage.detach() * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean() + entropy_loss
            # policy_loss = -(advantage.detach() * ratio).mean()
                        
            actor_optim.zero_grad()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(Actor.parameters(), 0.5) #Max Grad Norm
            actor_optim.step()
            

            value_loss = mse_loss(vals_s1.squeeze(-1), returns)
            critic_optim.zero_grad()
            value_loss.backward()
            # torch.nn.utils.clip_grad_norm_(Critic.parameters(), 0.5) #Max Grad Norm
            critic_optim.step()
            
            
            # total_loss = policy_loss + value_loss
            
            actor_loss_list.append(policy_loss.detach().cpu().numpy())
            critic_loss_list.append(value_loss.detach().cpu().numpy())
            
            
            # acer_optim.zero_grad()
            # total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(PolicyNet1.parameters(), 0.5) #Max Grad Norm
            # acer_optim.step()
            
            
            # for name, param in PolicyNet1.named_parameters():
            #     if( torch.any(torch.isnan(param)) ):
            #         print(name)
            #         print(param)
            #         BREAKPOINTBULLSHIT
            for name, param in Actor.named_parameters():
                if( torch.any(torch.isnan(param)) ):
                    print(name)
                    print(param)
                    BREAKPOINTBULLSHIT
            for name, param in Critic.named_parameters():
                if( torch.any(torch.isnan(param)) ):
                    print(name)
                    print(param)
                    BREAKPOINTBULLSHIT
                
            
            # print('---')
            # print('s1')
            # print(s1)
            print('a1')
            print(a1)
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
            
            
        
        # [vals, probs] = PolicyNet1(s2)
        with torch.no_grad():
            for _ in range(take_n_actions):
                s1, a1, r1, s2, d2, log_probs_old, returns = env.buffer.get_SARS()
                probs = Actor(s2)
                newest_probs = probs[:,0,0].view(-1,1)
                action_pd = torch.distributions.Normal(newest_probs, 0.05*torch.ones_like(newest_probs))
                next_actions = action_pd.sample()
                log_probs_sample = action_pd.log_prob(next_actions)
                env.step(next_actions, log_probs_sample, Critic)
                
                if(d2[0,0]):
                    env.render_init()
                env.render(0)
                # time.sleep(0.05)
                
            

    print(epoch)
    print('Policy Loss Avg: {}. Value Loss Avg: {}. Avg Returns: {}'.format(np.array(actor_loss_list).mean(), np.array(critic_loss_list).mean(), returns.mean()))
    
    with torch.no_grad():
        # Useful extra info
        approx_kl1 = ((torch.exp(ratio) - 1) - ratio).mean() #Stable Baselines 3
        approx_kl2 = (log_probs_old - log_probs).mean()    #Open AI Spinup
        # print('kl approx : {} : {} : {}'.format(approx_kl1, approx_kl2, ratio.mean()))
    
    # print(Qvals.reshape([siz,siz]))
    
        

#%%
import time
num_envs=2
test_env = CartPole(num_envs = num_envs, buf_horizon=10, rand_reset=False)
test_env.render_init()
#%%
env_ids = torch.arange(num_envs)
test_env.reset_idx(env_ids)

test_env.state[0,:] = 0
test_env.state[0,2] = 10*torch.pi/180.0
#%%

view_len = 3000
env_view_id = 0
for i in range(view_len):

    # print(i)
    
    test_env.render(env_view_id)
    s1, a1, r1, s2, d2, log_probs_old, returns = test_env.buffer.get_SARS() 
    
    probs = Actor(s2)
    newest_probs = probs[:,0,0].view(-1,1)
    action_pd = torch.distributions.Normal(newest_probs, 0.1*torch.ones_like(newest_probs))
    
    a = test_env.joy.get_axis()
    # next_actions = action_pd.sample()
    next_actions = probs[:,0,0].reshape(-1,1)
    log_probs = action_pd.log_prob(next_actions)
    
    test_env.step(next_actions, log_probs, Critic)
    
    print(next_actions[0,0])
    if(d2[0,0]):
        print('RESET')
    # print(newest_probs)

    

# env_render.close()
    
# #%%

# for name, param in PolicyNet2.named_parameters():
#         print(name)
#         print(param)

# #%%
# i = 0
# while(1):
#     i += 1
#     a = torch.distributions.Categorical(probs=probs[-2,:]).sample()
#     if(a != 0):
#         print('WOW')
#         print(i)
#         print(a)
#         break
   