# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:28:48 2023

@author: Plutonium
"""

import torch
import matplotlib.pyplot as plt
import pygame as py
import numpy as np 
from joystick import Joystick
import time


#%%
torch.set_default_device("cuda")

# define colors  
BLACK = (0 , 0 , 0)  
RED = (255 , 0 , 0)  
GREEN = (0 , 255 , 0)  
BLUE = (0 , 0 , 255)  


    
def RandTensorRange(size, min_val, max_val):
    tensor_range = max_val-min_val
    rt = torch.rand(size)*tensor_range + min_val
    return rt

class CartPole_4P():
    def __init__(self, num_envs=2, buf_horizon=10, gamma=0.9, rand_reset=False):
        self.dt = 0.01
        self.gravity = 9.81
        self.num_envs = num_envs
        self.buffer_hor = buf_horizon
        self.num_players = 4
        self.num_actions = 1*self.num_players
        self.num_states = 5

        self.rand_reset = rand_reset
        
        # ele 0 : Position
        # ele 1 : Velocity
        # ele 1 : Theta
        # ele 1 : Omega
        self.state = torch.zeros( (self.num_envs, self.num_players, self.num_states) ) 
        self.position = self.state[:,:,0].view(self.num_envs, self.num_players, 1)
        self.velocity = self.state[:,:,1].view(self.num_envs, self.num_players, 1)
        self.theta = self.state[:,:,2].view(self.num_envs, self.num_players, 1)
        self.omega = self.state[:,:,3].view(self.num_envs, self.num_players, 1)
        self.target = self.state[:,:,4].view(self.num_envs, self.num_players, 1)
        
        # self.kinematics_integrator = 'euler'
        self.kinematics_integrator = 'semi-euler'
        
        # Cart Variables
        self.min_cart_mass = 1.0
        self.max_cart_mass = 1.0
        self.cart_mass = RandTensorRange( (self.num_envs, 1, 1), self.min_cart_mass, self.max_cart_mass)
        # Pole Variables
        self.min_pole_mass = 0.1
        self.max_pole_mass = 0.1
        self.pole_mass = RandTensorRange( (self.num_envs, self.num_players, 1), self.min_pole_mass, self.max_pole_mass)
        self.length = 0.5 # Half Length
        self.polemass_precalc = self.pole_mass*self.length
        # self.Inertia = (1/3)*self.pole_mass*(self.length*2)**2
        
        self.total_mass = self.cart_mass + self.pole_mass

        self.force_scale = 25   #Scales actions from [-1, 1] -> [-fs, fs]

        
        # Angle which fails the episode
        self.theta_threshold_radians = 60 * 2 * torch.pi / 360
        # Position which fails the episode
        self.x_threshold = 2.4
        
        deg2rad = torch.pi/180.0
        self.rand_pos_range = 1.0               #Starting Position Max in m
        self.rand_vel_range = 1.0               #Starting Velocity Max in m/s
        self.rand_theta_range = 5.0*deg2rad    #Starting Angle Max in radians
        self.rand_omega_range = 1.0*deg2rad    #Starting Angular Vel in rad/s
        

        self.rand_target_range = self.x_threshold-0.5
        
        
        # Scale the observation returned by get_SARS
        self.pos_scale = 1.0/self.x_threshold
        self.vel_scale = 1.0/25.0
        self.theta_scale = 1/(torch.pi/2)
        self.omega_scale = 1.0/25.0
        self.target_scale = 1.0/(self.rand_target_range*2)
        self.state_scaler = torch.tensor([[ self.pos_scale, self.vel_scale, self.theta_scale, self.omega_scale, self.target_scale]])
        
        self.joy = Joystick()
        # self.render_init()
        
    def register_buffer(self, replay_buf):
        self.buffer=replay_buf
        
    def step(self, actions, log_probs, ValueNet):
        with torch.no_grad():
            
                
            force = self.force_scale * actions.view(self.num_envs, self.num_players, 1)
            
            self.costheta = torch.cos(self.theta)
            self.sintheta = torch.sin(self.theta)
            
            tmp = (force + self.polemass_precalc*self.omega**2 * self.sintheta)/self.total_mass
            alpha = (self.gravity * self.sintheta - self.costheta*tmp) / (self.length * (4.0/3.0 - self.pole_mass*self.costheta ** 2/self.total_mass))
            # alpha += -self.omega*0.1
            accel = tmp - self.pole_mass * alpha * self.costheta / self.total_mass
            
            # print(self.theta[0])
            # print(alpha)
            
            buf_states1 = self.state*self.state_scaler.view(1,1,self.num_states)
            buf_states1 = buf_states1.view(self.num_envs, self.num_states*self.num_players)

            self.buffer.update1(buf_states1, actions, log_probs)
            
            if self.kinematics_integrator == 'euler':
                dxdt = torch.cat( (self.velocity.view((-1,1)), accel, self.omega.view((-1,1)), alpha, torch.zeros_like(alpha)), dim=1)
                self.state[...] += dxdt*self.dt
                
            else: # semi-implicit euler
                self.velocity += accel*self.dt      #| - self.velocity*0.01
                self.position += self.velocity*self.dt
                self.omega += alpha*self.dt 
                self.theta += self.omega*self.dt
                
            
            self.out_of_bounds = (torch.where(torch.abs(self.position) > self.x_threshold, 1, 0)).squeeze()
            self.extream_angle = (torch.where(torch.abs(self.theta)  > self.theta_threshold_radians, 1, 0)).squeeze()
            self.done = (self.out_of_bounds | self.extream_angle)
            
                    
            self.reward_angle = (1.0-torch.abs(self.theta))**2.0
            self.reward_dist =  1.0-((self.position - self.target)/(self.x_threshold))**2.0
            self.reward = (self.reward_angle + self.reward_dist)/2.0
            self.reward = torch.where(self.done==1, -1.0, 1.0-torch.abs(actions))
            self.reward_summed = self.reward.sum(dim=1)
            self.done_summed = self.done.sum(dim=1)
            
            
            # Need to think about how to fix this in 4 player mode... 
            
            # self.reached_goal = ((self.position - self.target)**2 < 0.075) & (torch.abs(self.velocity) < 0.1)
            # env_ids = self.reached_goal.view(-1).nonzero(as_tuple=False).squeeze(-1)
            # if len(env_ids) > 0:
                # self.reset_goal(env_ids)
                # self.reward_summed[env_ids] = 100.0
                
                
                
            
            
            buf_states2 = self.state*self.state_scaler.view(1,1,self.num_states)
            buf_states2 = buf_states2.view(self.num_envs, self.num_states*self.num_players)
            
            [vals, probs] = ValueNet(buf_states2)
            
            self.buffer.update2(self.reward_summed, buf_states2, self.done_summed, vals)
    
            
            env_ids = self.done_summed.view(-1).nonzero(as_tuple=False).squeeze(-1)
    
            if len(env_ids) > 0:
                self.reset_idx(env_ids)
                
            

                    
            
            
    def reset_idx(self, env_ids):
        
        if(self.rand_reset):
            self.position[env_ids, ...] = RandTensorRange( (len(env_ids), self.num_players, 1), -self.rand_pos_range, self.rand_pos_range)
            self.velocity[env_ids, ...] = RandTensorRange( (len(env_ids), self.num_players, 1), -self.rand_vel_range, self.rand_vel_range)
            self.theta[env_ids, ...] = RandTensorRange( (len(env_ids), self.num_players, 1), -self.rand_theta_range, self.rand_theta_range)
            self.omega[env_ids, ...] = RandTensorRange( (len(env_ids), self.num_players, 1), -self.rand_omega_range, self.rand_omega_range)   
            self.target[env_ids, ...] = RandTensorRange( (len(env_ids), self.num_players, 1), -self.rand_target_range, self.rand_target_range)   
        else:
            self.position[env_ids, :] = torch.zeros( (len(env_ids), self.num_players, 1))
            self.velocity[env_ids, :] = torch.zeros( (len(env_ids), self.num_players, 1))
            self.theta[env_ids, :] = torch.zeros( (len(env_ids), self.num_players, 1))
            self.omega[env_ids, :] = torch.zeros( (len(env_ids), self.num_players, 1))   
            self.target[env_ids, :] = torch.zeros( (len(env_ids), self.num_players, 1))   
            
    def reset_goal(self, env_ids):
        self.target[env_ids, :] = RandTensorRange( (len(env_ids), 1), -self.rand_target_range, self.rand_target_range)
        
    def render_init(self):

        py.init()
        self.window_height = 500
        self.window_width = 1000
        self.window = py.display.set_mode((self.window_width, self.window_height))        
        
        py.display.init()

        
        self.clock = py.time.Clock()
        self.canvas = py.Surface((self.window_width, self.window_height))
        
        # define object sizes in pixels
        meters2pixels = 100
        self.rod_len = meters2pixels * self.length*2
        self.rod_thickness = self.rod_len/10
        self.cart_width = 1*meters2pixels
        self.cart_height = self.cart_width/5
        self.goal_size = 5
        self.cart_image = py.Surface((self.cart_width , self.cart_height))  
        self.rod_image  = py.Surface((self.rod_thickness , self.rod_len))  
        self.goal_image  = py.Surface((self.goal_size, self.goal_size))  
        # for making transparent background while rotating an image  
        self.cart_image.set_colorkey(BLACK)  
        self.rod_image.set_colorkey(BLACK)  
        self.goal_image.set_colorkey(BLACK)
        # fill the rectangle / surface with green color  
        self.cart_image.fill(BLUE)  
        self.rod_image.fill(GREEN)  
        self.goal_image.fill(RED)  
        # creating a copy of orignal image for smooth rotation  

        self.cart_copy = self.cart_image.copy()
        self.rod_copy = self.rod_image.copy()
        self.goal_copy = self.goal_image.copy()
        self.cart_copy.set_colorkey(BLACK)
        self.rod_copy.set_colorkey(BLACK)
        self.goal_copy.set_colorkey(BLACK)
        # define rect for placing the rectangle at the desired position  

        # screen center in pixels
        self.screen_center = (self.window_width/2, self.window_height/2)
        self.cart_center = (self.cart_width/2, self.cart_height/2)
        # scales the x travel limit to the edge of the screen
        self.pos_scale = (self.window_width-self.cart_width)/2/self.x_threshold
        
    def render(self, env_id):
        position = self.position[env_id, 0].detach().cpu().numpy() * self.pos_scale
        angle = -(self.theta[env_id, 0].detach().cpu().numpy() + torch.pi)
        goal_pos = self.target[env_id, 0].detach().cpu().numpy() * self.pos_scale
        
        # clear the screen every time before drawing new objects  
        self.window.fill(BLACK)  

        new_cart = self.cart_image
        cart_rect = new_cart.get_rect()

        cart_rect.center = (position + self.screen_center[0], self.screen_center[1])
        self.window.blit(new_cart, cart_rect)
                
        new_rod = py.transform.rotate(self.rod_image, angle*180.0/3.1415)
        rod_rect = new_rod.get_rect()  
        rod_rect.center = (self.rod_len/2*np.sin(angle) +  cart_rect.center[0], self.rod_len/2*np.cos(angle) + cart_rect.center[1])
        self.window.blit(new_rod, rod_rect)  
        
        new_goal = self.goal_image
        goal_rect = new_goal.get_rect()  
        goal_rect.center = (goal_pos + self.screen_center[0], self.screen_center[1])
        self.window.blit(new_goal, goal_rect)  
        
        py.display.flip()  
        py.event.pump();
        


    def close(self):
        py.quit()
            

            
            


# #%%
# from buffer import Buffer
# cp = CartPole_4P(num_envs = 2, buf_horizon=6, gamma=0.9, rand_reset=True)
# replay_buffer = Buffer(cp.buffer_hor, cp.num_envs, cp.num_actions, cp.num_states * cp.num_players, 0.99)
# cp.register_buffer(replay_buffer)

# cp.render_init()

# import torch.nn as nn
# class Policy(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # define actor and critic networks
        
#         n_features = 20
#         n_actions = 4
#         layer1_count = 256
#         layer2_count = 128
#         layer3_count = 64

        
#         self.shared1 = nn.Sequential(
#                                     nn.Linear(n_features, layer1_count),
#                                     nn.ELU()
#                                     )
        
#         self.shared2 = nn.Sequential(
#                                     nn.Linear(layer1_count+n_features, layer2_count),
#                                     nn.ELU()
#                                     )
        
        
#         self.policy1 = nn.Sequential(
#                                     nn.Linear(layer2_count+n_features, layer3_count),
#                                     nn.ELU()
#                                     )
#         self.policy2 = nn.Sequential(
#                                     nn.Linear(layer3_count+n_features, n_actions),
#                                     nn.Tanh(),
#                                     )
        
#         self.value1 = nn.Sequential(
#                                     nn.Linear(layer2_count+n_features, layer3_count),
#                                     nn.ELU()
#                                     )
#         self.value2 = nn.Sequential(
#                                     nn.Linear(layer3_count+n_features, 1),
#                                     )

#     def forward(self, x):
#         s1 = torch.cat((self.shared1(x), x), dim=-1)
#         s2 = torch.cat((self.shared2(s1), x), dim=-1)
#         v1 = torch.cat((self.value1(s2) , x), dim=-1)
#         v2 = self.value2(v1) 
#         p1 = torch.cat((self.policy1(s2), x), dim=-1)
#         p2 = self.policy2(p1)        
#         return v2, p2

# policy = Policy()
# #%%

# pos_log = []
# vel_log = []
# theta_log = []
# omega_log = []
# reward_log = []

# env_2_watch = 0;

# cp.position[...] = 0
# cp.velocity[...] = 0
# cp.theta[...] = 0*torch.pi/180.0
# cp.omega[...] = 0

# cp.theta[...] = 1*torch.pi/180.0
# #%%

# for _ in range(10):
    
#     start_time = time.time()
#     pos_log.append(cp.state[env_2_watch, 0].detach().cpu().numpy())
#     vel_log.append(cp.state[env_2_watch, 1].detach().cpu().numpy())
#     theta_log.append(cp.state[env_2_watch, 2].detach().cpu().numpy())
#     omega_log.append(cp.state[env_2_watch, 3].detach().cpu().numpy())
    
    
#     cp.render(env_2_watch)
#     a = cp.joy.get_axis()
#     # actions = torch.ones((cp.num_envs,1))*a[0]
#     actions = torch.ones((cp.num_envs, cp.num_actions))*0.5
#     log_probs = torch.ones((cp.num_envs, cp.num_actions))
#     cp.step(actions, log_probs, policy)
#     reward_log.append(cp.reward[env_2_watch].detach().cpu().numpy())
#     # print(cp.reward[0,...])
#     elapsed_time = time.time() - start_time
#     # print('Elapsed Time : {}'.format(elapsed_time))
#     # print('FPS : {}'.format(cp.num_envs/elapsed_time))

    
#     [s1,a1,r1,s2,d, log_probs_old, returns] = cp.buffer.get_SARS()
#     # print('s1')
#     # print(s1[0, : , 0])
#     # print('s2')
#     # print(s2[0, :, 0])
#     # print('a1')
#     # print(a1)
#     # print('r')
#     # print(r1)
#     # print('d')
#     # print(d)
#     # print('advantage')
#     # print(cp.buffer.rewards_to_go)

#     cp.render(0)
    

# # plt.figure(figsize=(9, 3))
# # plt.subplot(321)
# # plt.grid(True)
# # plt.plot(pos_log)
# # plt.subplot(322)
# # plt.plot(vel_log)
# # plt.grid(True)
# # plt.subplot(323)
# # plt.plot(theta_log)
# # plt.grid(True)
# # plt.subplot(324)
# # plt.plot(omega_log)
# # plt.grid(True)
# # plt.subplot(325)
# # plt.grid(True)
# # plt.plot(reward_log)


# #%%

# [s1,a1,r1,s2,d, lp_old, returns] = cp.buffer.get_SARS()
# print('s1')
# print(s1)
# print('s2')
# print(s2)
# print('a1')
# print(a1)
# print('r')
# print(r1)
# print('d')
# print(d)
# print('rewards_to_go')
# print(cp.buffer.rewards_to_go)

# #%%

# # [s1,a1,r1,s2,d] = cp.buffer.get_SARS_minibatch(6)
# # print('s1')
# # print(s1)
# # print('s2')
# # print(s2)
# # print('a1')
# # print(a1)
# # print('r')
# # print(r1)
# # print('d')
# # print(d)

# #%%

# dones_tmp = cp.buffer.d.clone()
# dones_tmp[:,0] = False
# dones_mask = torch.where(dones_tmp, 0, 1)
# dones_mask = torch.cumprod(dones_mask, dim=1)
# print(dones_mask)
