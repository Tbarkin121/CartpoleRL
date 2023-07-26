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
from cartpole_discrete import CartPole
import time

num_envs =   2
horizon = 3
gamma = 0.9
num_epochs = 2500
num_q_holds = 20
minibatch_steps = 1
minibatch_size = 1000
take_n_actions = 1
entropy_coff_inital = 0.0
clip_range = 0.1
            
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
        
        actor_std_layers = [
                        nn.Linear(128, n_actions),
                    ]
        
        
        self.critic = nn.Sequential(*critic_layers)
        self.actor_top = nn.Sequential(*actor_top_layers)
        self.actor_mean = nn.Sequential(*actor_mean_layers)
        self.actor_std = nn.Sequential(*actor_std_layers)


    def forward(self, x):
        state_values = self.critic(x)  # shape: [n_envs,]
        a0 = self.actor_top(x)  # shape: [n_envs, n_actions]
        mean = self.actor_mean(a0)
        std = self.actor_std(a0) + 0.01
        action_distribution_vec = torch.cat((mean, std), dim=2)
        
        return (state_values, mean)

    
#%%
a = torch.rand([3,5])
PolicyNet1 = Policy()

# critic_optim = optim.Adam(PolicyNet1.critic.parameters(), lr=1e-3)

# actor_optim = optim.Adam( [ { 'params': PolicyNet1.actor_top.parameters() },
#                             { 'params': PolicyNet1.actor_mean.parameters() },
#                             { 'params': PolicyNet1.actor_std.parameters()} ], 
#                             lr=1e-3)


# acer_optim = optim.Adam( [  { 'params': PolicyNet1.critic.parameters() },
#                             { 'params': PolicyNet1.actor_top.parameters() },
#                             { 'params': PolicyNet1.actor_mean.parameters() },
#                             { 'params': PolicyNet1.actor_std.parameters()} ], 
#                             lr=1e-3)
                             
                             
acer_optim = optim.Adam(PolicyNet1.parameters(), lr=1e-3)

PolicyNet2 = Policy()
PolicyNet2.eval()

mse_loss = torch.nn.MSELoss()
env = CartPole(num_envs = num_envs, buf_horizon=horizon, gamma=gamma)
log_probs = torch.zeros((num_envs, horizon))
log_probs_old = torch.zeros((num_envs, horizon)).detach()
# env.render_init()
#%%




# for epoch in range(1):
for epoch in range(num_epochs):

    # PolicyNet2.load_state_dict(PolicyNet1.state_dict())
    actor_loss_list = []
    critic_loss_list = []
    
    # for i in range(1):
    for i in range(num_q_holds):
        # s1, a1, r1, s2, d2 = env.buffer.get_SARS() 
        
        for _ in range(1):
        # for _ in range(minibatch_steps):
            # s1, a1, r1, s2, d2 = env.buffer.get_SARS_minibatch(minibatch_size) 
            s1, a1, r1, s2, d2 = env.buffer.get_SARS()
            
            # print('---')
            # print('s1')
            # print(s1[0:1,...])
            # print('a1')
            # print(a1[0:1,...])
            # print('r1')
            # print(r1[0:1,...])
            # print('s2')
            # print(s2[0:1,...])
            # print('d2')
            # print(d2[0:1,...])
            # print('env.buffer.value_gamma_scaler[0:1,...]')
            # print(env.buffer.value_gamma_scaler[0:1,...])
            
            [vals_s1, probs_s1] = PolicyNet1(s1)
            [vals_s2, probs_s2] = PolicyNet2(s2)
            
            
            action_pd_s1 = torch.distributions.Categorical( logits=probs_s1 )
            action_pd_s2 = torch.distributions.Categorical( logits=probs_s2 )

            
           

            
            returns = env.buffer.value_gamma_scaler*vals_s2[:,0] + r1
            
            # td_error = returns - vals_s1.squeeze(-1)
            advantage = returns - vals_s1.squeeze(-1)

            # # normalize advantage... (Doesn't Seem to work)
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
        
            # prob_act = probs.gather(1, actions.reshape([-1,1]))
            # log_probs = torch.log(prob_act)
            log_probs = action_pd_s1.log_prob(a1)
            ratio = torch.exp(log_probs - log_probs_old)
            # print('log_probs')
            # print(log_probs)
            # print('log_probs_old')
            # print(log_probs_old)
            log_probs_old = log_probs.clone().detach()
            


    
            policy_loss_1 = advantage * ratio
            policy_loss_2 = advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            # policy_loss = -policy_loss_2.mean()
            value_loss = advantage.pow(2).mean()
            
            
            # entropy_coff = entropy_coff_inital * (1-epoch/num_epochs)
            entropy_coff = entropy_coff_inital
            entropy_loss = -action_pd_s1.entropy().mean() * entropy_coff
            
            
            total_loss = policy_loss + value_loss + entropy_loss
            
            actor_loss_list.append(policy_loss.detach().cpu().numpy())
            critic_loss_list.append(value_loss.detach().cpu().numpy())
            
            
            acer_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(PolicyNet1.parameters(), 0.5) #Max Grad Norm
            acer_optim.step()
            
            PolicyNet2.load_state_dict(PolicyNet1.state_dict())
            
            for name, param in PolicyNet1.named_parameters():
                if( torch.any(torch.isnan(param)) ):
                    print(name)
                    print(param)
                    BREAKPOINTBULLSHIT
            log_probs_old_backup = log_probs.clone()
            
        
        [vals, probs] = PolicyNet1(env.buffer.s2)
        action_pd = torch.distributions.Categorical( logits=probs )
        # for _ in range(1):
        for _ in range(take_n_actions):
            # env.render(0)
            next_actions = action_pd.sample()[:, 0].reshape(-1,1)
            env.step(next_actions)
            

    print(epoch)
    print('Actor Loss Avg: {}. Critic Loss AVg: {}'.format(np.array(actor_loss_list).mean(), np.array(critic_loss_list).mean()))
    # print(Qvals.reshape([siz,siz]))
    
        

#%%
import time
num_envs=2
test_env = CartPole(num_envs = num_envs, buf_horizon=10)
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
    s1, a1, r1, s2, d2 = test_env.buffer.get_SARS() 
    
    [val, probs] = PolicyNet1(s1)
    action_pd = torch.distributions.Categorical( logits=probs )
    a = test_env.joy.get_axis()
    next_actions = action_pd.sample()[:, 0].view(-1,1) + a[0]

    test_env.step(next_actions)
    print(next_actions[0,0])
    # time.sleep(1/120)
    
    if(view_len <=100):
        # print(s1[-1,0,...])
        print(test_env.buffer.r[-1,0])
        # print(test_env.buffer.d[-1,0])
        print('Step : {}. Action : {}'.format(i, next_actions[0]))

    

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
   