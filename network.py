import torch
import torch.nn as nn
import torch.nn.functional as f

# for initalising all layers of a network at once
def initialize_weights(net, low=-3e-2, high=3e-2):
    for param in net.parameters():
        param.data.uniform_(low, high)

# the actor takes a state and outputs an estimated best action
class Actor(nn.Module):
    def __init__(self, action_size, state_size, hidden_in_size, hidden_out_size):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size,hidden_in_size)
        self.fc2 = nn.Linear(hidden_in_size,hidden_out_size)
        self.fc3 = nn.Linear(hidden_out_size,action_size)
        initialize_weights(self)


    def forward(self, state):
        layer_1 = f.relu(self.fc1(state))
        layer_2 = f.relu(self.fc2(layer_1))
        action = torch.tanh(self.fc3(layer_2)) # tanh because the action space is -1 to 1
        return action

# the critic takes the states and actions of both agents and outputs a prob distribution over estimated Q-values
class Critic(nn.Module):
    def __init__(self, actions_size, states_size, hidden_in_size, hidden_out_size, num_atoms): # num_atoms is the granularity of the bins
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(states_size,hidden_in_size)
        self.fc2 = nn.Linear(hidden_in_size+actions_size,hidden_out_size) # the actions are added to the second layer
        self.fc3 = nn.Linear(hidden_out_size,num_atoms)
        initialize_weights(self)


    def forward(self, states, actions, log=False): # log determines whether the softmax or log softmax is outputed for critic updates
        layer_1 = f.relu(self.fc1(states))
        layer_1_cat = torch.cat([layer_1, actions], dim=1)
        layer_2 = f.relu(self.fc2(layer_1_cat))
        Q_probs = self.fc3(layer_2)
        if log:
            return f.log_softmax(Q_probs, dim=-1)
        else:
            return f.softmax(Q_probs, dim=-1) # softmax converts the Q_probs to valid probabilities (i.e. 0 to 1 and all sum to 1)