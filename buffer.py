from collections import deque
import random
from utilities import transpose_list
import numpy as np


class ReplayBuffer:
    def __init__(self,size, n_steps = 5, discount_rate=0.99):
        self.size = size
        self.n_steps = n_steps
        self.discount_rate = discount_rate
        self.deque = deque(maxlen=self.size)                        # this holds all the experience to be sampled for training
        self.n_step_deque = deque(maxlen=self.n_steps)              # new experience first goes here until n timesteps have passed     

    def push(self,transition):
        """push into the buffer"""
        
        self.n_step_deque.append(transition)
        if len(self.n_step_deque)==self.n_steps:             # once the deque has n steps we can start accumilating rewards
            start_obs, start_action, start_reward, start_next_obs, start_done = self.n_step_deque[0]  # first experience
            n_obs, n_action, n_reward, n_next_obs, n_done = self.n_step_deque[-1] # last experience
            
            summed_reward = np.zeros(2)                                    # initialise
            for i,n_transition in enumerate(self.n_step_deque):            # for each experience
                obs, action, reward, next_obs, done = n_transition         
                summed_reward += reward*self.discount_rate**(i+1)          # accumulate rewards
                if np.any(done):                                           # stop if done
                    break
            
            transition = [start_obs, start_action, summed_reward, n_next_obs, n_done] # we take first obs and action, summed rewards and last next obs and done
            self.deque.append(transition) # 
        

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize) # sample a minibatch from buffer

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)
    
    def reset(self):
        self.n_step_deque = deque(maxlen=self.n_steps) # reset n-step deque between episodes