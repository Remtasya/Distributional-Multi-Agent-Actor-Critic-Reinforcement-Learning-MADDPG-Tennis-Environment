# individual network settings for each actor + critic pair

from network import Actor, Critic
from utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np

from noise import OUNoise, BetaNoise, GaussNoise, WeightedNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

class DDPGAgent:
    def __init__(self, action_size, state_size, hidden_in_size, hidden_out_size, num_atoms, lr_actor, lr_critic, l2_decay = 0.0001, noise_type = 'BetaNoise'):
        super(DDPGAgent, self).__init__()

        # creating actors, critics and targets using the specified layer sizes. Note for the critics we assume 2 agents
        self.actor = Actor(action_size, state_size, hidden_in_size, hidden_out_size).to(device)
        self.critic = Critic(2*action_size, 2*state_size, hidden_in_size, hidden_out_size, num_atoms).to(device)
        self.target_actor =  Actor(action_size, state_size, hidden_in_size, hidden_out_size).to(device)
        self.target_critic = Critic(2*action_size, 2*state_size, hidden_in_size, hidden_out_size, num_atoms).to(device)
        self.noise_type = noise_type
        
        if noise_type=='OUNoise': # if we're using OUNoise it needs to be initialised as it is an autocorrelated process
            self.noise = OUNoise(action_size)
            
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

       # initialize optimisers using specigied learning rates
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor, weight_decay=l2_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=l2_decay)

    def act(self, obs, noise_scale=0.0):
        obs = obs.to(device)
        action = self.actor(obs)
        
        if noise_scale==0.0: # if no noise then just return action as is
            pass
        elif self.noise_type=='OUNoise':
            noise = 1.5*self.noise.noise()
            action += noise_scale*(noise-action)
            action = torch.clamp(action,-1,1)
        elif self.noise_type=='BetaNoise':
            action = BetaNoise(action, noise_scale)
        elif self.noise_type=='GaussNoise':
            action = GaussNoise(action, noise_scale)
        elif self.noise_type=='WeightedNoise':
            action = WeightedNoise(action, noise_scale)
        return action
    
    # target actor is only used for updates not exploration and so has no noise
    def target_act(self, obs):
        obs = obs.to(device)
        action = self.target_actor(obs)
        return action