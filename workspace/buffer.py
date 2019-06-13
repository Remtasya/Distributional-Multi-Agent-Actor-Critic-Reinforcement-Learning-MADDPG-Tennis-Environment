from collections import deque
import random
from utilities import transpose_list
import numpy as np


class ReplayBuffer:
    def __init__(self,size, n_steps = 5, discount_rate=0.99):
        self.size = size
        self.n_steps = n_steps
        self.discount_rate = discount_rate
        self.deque = deque(maxlen=self.size)
        self.n_step_deque = deque(maxlen=self.n_steps)

    def push(self,transition):
        """push into the buffer"""
        
        self.n_step_deque.append(transition)
        if len(self.n_step_deque)==5:
            start_obs, start_action, start_reward, start_next_obs, start_done = self.n_step_deque[0]
            n_obs, n_action, n_reward, n_next_obs, n_done = self.n_step_deque[-1]
            
            summed_reward = np.zeros(2)
            for i,n_transition in enumerate(self.n_step_deque):
                obs, action, reward, next_obs, done = n_transition
                summed_reward += reward*self.discount_rate**(i+1)
                if np.any(done):
                    break
            
            transition = [start_obs, start_action, summed_reward, n_next_obs, n_done]
            self.deque.append(transition)
        

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)
    
    def reset(self):
        self.n_step_deque = deque(maxlen=self.n_steps)