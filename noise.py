import numpy as np
import torch


# OUNoise from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_size, mu, theta, sigma):
        self.action_size = action_size
        self.mu = mu                                                # mu - the long-term mean
        self.theta = theta                                          # theta - the mean reversion strength
        self.sigma = sigma                                          # sigma - the noise magnitude
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x)) # mean reversion plus noise
        self.state = x + dx
        return torch.tensor(self.state).float()
    
def BetaNoise(action, noise_scale):
    
    action = action.detach().numpy()                    # since the input is a tensor we must convert it to numpy before operations
    sign = np.sign(action)                              # tracking the sign so we can flip the samples later
    action = abs(action)                                # we only use right tail of beta
    alpha = 1/noise_scale                               # this determines the how contentrated the beta dsn is
    value = 0.5+action/2                                # converting from action space of -1 to 1 to beta space of 0 to 1
    beta = alpha*(1-value)/value                        # calculating beta
    beta = beta + 1.0*((alpha-beta)/alpha)              # adding a little bit to beta prevents actions getting stuck at -1 or 1
    sample = np.random.beta(alpha, beta)                # sampling from the beta distribution
    sample = sign*sample+(1-sign)/2                     # flipping sample if sign is <0 since we only use right tail of beta dsn
                    
    action_output = 2*sample-1                          # converting back to action space -1 to 1
    return torch.tensor(action_output)                  # converting back to tensor

def GaussNoise(action, noise_scale):

    n = np.random.normal(0, 1, len(action))                                     # create some standard normal noise
    return torch.clamp(action+torch.tensor(noise_scale*n).float(),-1,1)         # add the noise to the actions

def WeightedNoise(action, noise_scale, action_type):
    """
    Returns the epsilon scaled noise distribution for adding to Actor
    calculated action policy.
    """
    if action_type=='continuous':
        target = np.random.uniform(-1,1,2)     # the action space is -1 to 1
    elif action_type=='discrete':
        target = np.random.uniform(0,1,4)      # action space is discrete
        target = target/sum(target)
    action = noise_scale*target+(1-noise_scale)*action.detach().numpy()      # take a weighted average with noise_scale as the noise weight
    return torch.tensor(action).float()