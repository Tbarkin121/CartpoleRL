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

class Buffer():
    def __init__(self, buf_horizon, num_envs, num_actions, num_states, gamma):
        self.buf_hor = buf_horizon
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.gamma_mask = torch.ones((self.buf_hor+1)) * self.gamma
        self.gamma_mask = torch.cumprod(self.gamma_mask, dim=0)/self.gamma
        
        # self.s1= torch.zeros([self.buf_len, self.num_states])
        # self.a = RandTensorRange([self.buf_len,], -1.0, 1.0)
        # self.r= torch.zeros([self.buf_len,])
        # self.s2 = torch.zeros([self.buf_len,self.num_states])
        # self.d = torch.zeros([self.buf_len,])==1
        
        self.s1 = torch.zeros([self.num_envs, self.buf_hor, self.num_states])
        self.a = torch.zeros([self.num_envs, self.buf_hor,])
        self.r = torch.zeros([self.num_envs, self.buf_hor,])
        self.s2 = torch.zeros([self.num_envs, self.buf_hor, self.num_states])
        self.d = torch.zeros([self.num_envs, self.buf_hor,]) == 1
        
        self.rewards_to_go = torch.zeros([self.num_envs, self.buf_hor,])
        self.value_gamma_scaler = torch.ones([self.num_envs, self.buf_hor,]) * self.gamma_mask[1:]

    def fill(self):
        pass
        # for i in range(self.buf_hor):
        #     actions = RandTensorRange([self.num_envs,], -1.0, 1.0)
        #     self.update(actions)
            
    def update1(self, s1, a1):
        with torch.no_grad():
            self.s1 = self.s1.roll(1, 1)
            self.s1[:, 0, :] = s1
            
            self.a = self.a.roll(1, 1)
            self.a[:, 0] = a1.view(-1)
            
            
    def update2(self, r2, s2, d2):
        with torch.no_grad():
            self.r = self.r.roll(1, 1)
            self.r[:, 0] = r2.view(-1)
            self.s2 = self.s2.roll(1, 1)
            self.s2[:, 0, :] = s2
            self.d = self.d.roll(1, 1)
            self.d[:, 0] = d2.view(-1)
            
            
            
            dones_tmp = self.d.clone()
            dones_tmp[:,0] = False
            
            dones_mask = torch.where(dones_tmp, 0, 1)
            dones_mask = torch.cumprod(dones_mask, dim=1)
            
            self.rewards_to_go = self.rewards_to_go.roll(1, 1)
            self.rewards_to_go[:, 0] = 0
            self.rewards_to_go += dones_mask*self.gamma_mask[0:self.buf_hor]*r2
            
            dones_mask2 = torch.where(self.d, 0, 1)
            dones_mask2 = torch.cumprod(dones_mask2, dim=1)
            self.value_gamma_scaler = dones_mask2*self.gamma_mask[1:]
            
        
    def get_SARS(self):
       # return self.s1, self.a, self.r, self.s2, self.d 
       return self.s1, self.a, self.rewards_to_go, self.s2, self.d 
   
    def get_SARS_minibatch(self, num_samples):
        env_ids = torch.randint(low=0, high=self.num_envs, size=(num_samples,))
        # return self.s1[env_ids, :, :], self.a[env_ids, :], self.r[env_ids, :], self.s2[env_ids, :, :], self.d[env_ids, :] 
        return self.s1[env_ids, :, :], self.a[env_ids, :], self.rewards_to_go[env_ids, :], self.s2[env_ids, :, :], self.d[env_ids, :] 
    
    
class CartPole():
    def __init__(self, num_envs=2, buf_horizon=10, gamma=0.9):
        self.dt = 0.01
        self.gravity = 9.81
        self.num_envs = num_envs
        self.buffer_len = num_envs * buf_horizon
        self.buffer_hor = buf_horizon
        self.num_actions = 1
        self.num_states = 5
        self.buffer = Buffer(self.buffer_hor, self.num_envs, self.num_actions, self.num_states, gamma)
        
        # ele 0 : Position
        # ele 1 : Velocity
        # ele 1 : Theta
        # ele 1 : Omega
        self.state = torch.zeros( (self.num_envs, self.num_states) ) 
        self.position = self.state[:,0].view(-1,1)
        self.velocity = self.state[:,1].view(-1,1)
        self.theta = self.state[:,2].view(-1,1)
        self.omega = self.state[:,3].view(-1,1)
        self.target = self.state[:,4].view(-1,1)
        
        # self.kinematics_integrator = 'euler'
        self.kinematics_integrator = 'semi-euler'
        
        # Cart Variables
        self.min_cart_mass = 1.0
        self.max_cart_mass = 1.0
        self.cart_mass = RandTensorRange( (self.num_envs, 1), self.min_cart_mass, self.max_cart_mass)
        # Pole Variables
        self.min_pole_mass = 0.1
        self.max_pole_mass = 0.1
        self.pole_mass = RandTensorRange( (self.num_envs, 1), self.min_pole_mass, self.max_pole_mass)
        self.length = 0.5 # Half Length
        self.polemass_precalc = self.pole_mass*self.length
        # self.Inertia = (1/3)*self.pole_mass*(self.length*2)**2
        
        self.total_mass = self.cart_mass + self.pole_mass

        self.force_scale = 10   #Scales actions from [-1, 1] -> [-fs, fs]

        
        # Angle which fails the episode
        self.theta_threshold_radians = 12 * 2 * torch.pi / 360
        # Position which fails the episode
        self.x_threshold = 2.4
        
        deg2rad = torch.pi/180.0
        # self.rand_pos_range = 5.0               #Starting Position Max in m
        # self.rand_vel_range = 2.0               #Starting Velocity Max in m/s
        # self.rand_theta_range = 10.0*deg2rad    #Starting Angle Max in radians
        # self.rand_omega_range = 2.0*deg2rad    #Starting Angular Vel in rad/s
        self.rand_pos_range = 0.0               #Starting Position Max in m
        self.rand_vel_range = 0.0               #Starting Velocity Max in m/s
        self.rand_theta_range = 0.0*deg2rad    #Starting Angle Max in radians
        self.rand_omega_range = 0.0*deg2rad    #Starting Angular Vel in rad/s

        self.rand_target_range = 4.0
        
        
        # Scale the observation returned by get_SARS
        self.pos_scale = 1.0/self.x_threshold
        self.vel_scale = 1.0/25.0
        self.theta_scale = 1/(torch.pi/2)
        self.omega_scale = 1.0/25.0
        self.target_scale = 1.0/(self.rand_target_range*2)
        self.state_scaler = torch.tensor([[ self.pos_scale, self.vel_scale, self.theta_scale, self.omega_scale, self.target_scale]])
        
        self.joy = Joystick()
        # self.render_init()
        
    def step(self, actions):
        with torch.no_grad():

            force = self.force_scale * actions
            
            self.costheta = torch.cos(self.theta)
            self.sintheta = torch.sin(self.theta)
                
            tmp = (force + self.polemass_precalc*self.omega**2 * self.sintheta)/self.total_mass
            alpha = (self.gravity * self.sintheta - self.costheta*tmp) / (self.length * (4.0/3.0 - self.pole_mass*self.costheta ** 2/self.total_mass))
            # alpha += -self.omega*0.1
            accel = tmp - self.pole_mass * alpha * self.costheta / self.total_mass
            
            # print(self.theta[0])
            # print(alpha)
            
            
            self.buffer.update1(self.state*self.state_scaler, actions)
            
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
            self.done = (self.out_of_bounds | self.extream_angle).view(-1,1)
            
                    
            self.reward_angle = (1.0-torch.abs(self.theta))**2.0
            self.reward_dist =  1.0-((self.position - self.target)/(self.x_threshold))**2.0
            self.reward = (self.reward_angle + self.reward_dist/5.0)/2.0
            
            self.reward = torch.where(self.done==1, -1.0, 1.0)
            
            self.buffer.update2(self.reward, self.state*self.state_scaler, self.done)
    
            
            env_ids = self.done.view(-1).nonzero(as_tuple=False).squeeze(-1)
    
            if len(env_ids) > 0:
                self.reset_idx(env_ids)
                    
            
            
    def reset_idx(self, env_ids):
        self.position[env_ids, :] = RandTensorRange( (len(env_ids), 1), -self.rand_pos_range, self.rand_pos_range)
        self.velocity[env_ids, :] = RandTensorRange( (len(env_ids), 1), -self.rand_vel_range, self.rand_vel_range)
        self.theta[env_ids, :] = RandTensorRange( (len(env_ids), 1), -self.rand_theta_range, self.rand_theta_range)
        self.omega[env_ids, :] = RandTensorRange( (len(env_ids), 1), -self.rand_omega_range, self.rand_omega_range)   
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
        position = self.position[env_id].detach().cpu().numpy() * self.pos_scale
        angle = -(self.theta[env_id].detach().cpu().numpy() + torch.pi)
        goal_pos = self.target[env_id].detach().cpu().numpy() * self.pos_scale
        
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
        


    def close(self):
        py.quit()
            

            
            
# #%%
# cp = CartPole(num_envs = 1, buf_horizon=10, gamma=0.9)
# cp.render_init()
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

# #%%
# cp.theta[...] = 1*torch.pi/180.0
# for _ in range(125):
    
#     start_time = time.time()
#     pos_log.append(cp.state[env_2_watch, 0].detach().cpu().numpy())
#     vel_log.append(cp.state[env_2_watch, 1].detach().cpu().numpy())
#     theta_log.append(cp.state[env_2_watch, 2].detach().cpu().numpy())
#     omega_log.append(cp.state[env_2_watch, 3].detach().cpu().numpy())
    
    
#     cp.render(env_2_watch)
#     a = cp.joy.get_axis()
#     # actions = torch.ones((cp.num_envs,1))*a[0]
#     actions = torch.zeros((cp.num_envs, 1))
#     cp.step(actions)
#     reward_log.append(cp.reward[env_2_watch].detach().cpu().numpy())
#     print(cp.reward[0,...])
#     elapsed_time = time.time() - start_time
#     # print('Elapsed Time : {}'.format(elapsed_time))
#     # print('FPS : {}'.format(cp.num_envs/elapsed_time))

    
#     [s1,a1,r1,s2,d] = cp.buffer.get_SARS()
#     print('s1')
#     print(s1)
#     print('s2')
#     print(s2)
#     print('a1')
#     print(a1)
#     print('r')
#     print(r1)
#     print('d')
#     print(d)
#     print('advantage')
#     print(cp.buffer.rewards_to_go)

    

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

# [s1,a1,r1,s2,d] = cp.buffer.get_SARS()
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
