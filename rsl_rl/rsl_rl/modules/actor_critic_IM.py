# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional

def reparameterize( mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def vae_loss(recon_x, x, mu, logvar):
    MSE = torch.nn.functional.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

class ActorCriticRAC(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        horizon = 1,
                        measurement = 1,
                        constraint_num = 1,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticRAC, self).__init__()
          
        activation = get_activation(activation)
         
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        self.one_obs_dims = int((mlp_input_dim_a)/horizon)
        self.hf_dims = measurement
         
        # Actor context model
        self.context_hidden_dims = 3 + 32
        contact_predictor_layer=[]
        contact_predictor_dim = [256,256]
        contact_predictor_layer.append(nn.Linear(mlp_input_dim_a, contact_predictor_dim[0]))
        contact_predictor_layer.append(activation)
        for l in range(len(contact_predictor_dim)):
            if l == len(contact_predictor_dim) - 1:
                contact_predictor_layer.append(nn.Linear(contact_predictor_dim[l], self.context_hidden_dims))
            else:
                contact_predictor_layer.append(nn.Linear(contact_predictor_dim[l], contact_predictor_dim[l + 1]))
                contact_predictor_layer.append(activation)
        self.context_model = nn.Sequential(*contact_predictor_layer)
         
        # critic context model
        contact_predictor_layer=[]
        contact_predictor_dim = [512,256,128]
        contact_predictor_layer.append(nn.Linear(mlp_input_dim_c, contact_predictor_dim[0]))
        contact_predictor_layer.append(activation)
        for l in range(len(contact_predictor_dim)):
            if l == len(contact_predictor_dim) - 1:
                contact_predictor_layer.append(nn.Linear(contact_predictor_dim[l], constraint_num))
            else:
                contact_predictor_layer.append(nn.Linear(contact_predictor_dim[l], contact_predictor_dim[l + 1]))
                contact_predictor_layer.append(activation)
        self.cost_critic = nn.Sequential(*contact_predictor_layer)
         
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.one_obs_dims + self.context_hidden_dims, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
         
        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
         
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
         
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
         
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)
     
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
     
    def reset(self, dones=None):
        pass
     
    def forward(self):
        raise NotImplementedError
     
    @property
    def action_mean(self):
        return self.distribution.mean
     
    @property
    def action_std(self):
        return self.distribution.stddev
     
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
     
    def update_distribution(self, observations):
        # no need to update context model when PPO updating.
        with torch.no_grad():
            context_hiddens = self.context_model(observations[...])
            vel,hidden = context_hiddens[...,:3],context_hiddens[...,3:]
            hidden = torch.nn.functional.normalize(hidden,2,dim=-1)
        mean = self.actor(torch.cat([observations[..., :self.one_obs_dims], vel, hidden], dim=-1))
        self.distribution = Normal(mean, mean*0. + self.std)
     
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
       
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
     
    def act_inference(self, observations):
        context_hiddens = self.context_model(observations[...])
        vel,hidden = context_hiddens[...,:3],context_hiddens[...,3:]
        hidden = torch.nn.functional.normalize(hidden,2,dim=-1)
        actions_mean = self.actor(torch.cat([observations[..., :self.one_obs_dims], vel, hidden], dim=-1))
        return actions_mean
     
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
     
    def evaluate_cost(self, critic_observations, **kwargs):
        return self.cost_critic(critic_observations)
     
    # update context model. 
    def update_context(self, obs, next_obs, next_next_obs, critic_obs):
        # update context model using RAC Loss
        context_hiddens = self.context_model(obs[...])
        vel,hidden = context_hiddens[...,:3],context_hiddens[...,3:]
        hidden = torch.nn.functional.normalize(hidden,2,dim=-1)

        with torch.no_grad():
            context_hidden_next = self.context_model(next_obs[...])
            _,hidden1 = context_hidden_next[...,:3],context_hidden_next[...,3:]
            hidden1 = torch.nn.functional.normalize(hidden1,2,dim=-1)
            context_hidden_next_next = self.context_model(next_next_obs[...])
            _,hidden2 = context_hidden_next_next[...,:3],context_hidden_next_next[...,3:]
            hidden2 = torch.nn.functional.normalize(hidden2,2,dim=-1)
        Loss_IM = ((hidden - hidden1).norm(p=2,dim=-1).mean() - 0.5*(hidden - hidden2).norm(p=2,dim=-1).mean()).clip(min=0.)
         
        loss_velovity = torch.nn.functional.mse_loss(vel , critic_obs[...,:3].detach())
        return Loss_IM, loss_velovity 
         
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
