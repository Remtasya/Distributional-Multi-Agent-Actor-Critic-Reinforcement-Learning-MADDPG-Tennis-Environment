import torch
import torch.nn as nn
import torch.nn.functional as f

def initialize_weights(net, low=-3e-2, high=3e-2):
    for param in net.parameters():
        param.data.uniform_(low, high)

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
        action = torch.tanh(self.fc3(layer_2))
        return action

class Critic(nn.Module):
    def __init__(self, actions_size, states_size, hidden_in_size, hidden_out_size):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(states_size,hidden_in_size)
        self.fc2 = nn.Linear(hidden_in_size+actions_size,hidden_out_size)
        self.fc3 = nn.Linear(hidden_out_size,1)
        initialize_weights(self)


    def forward(self, states, actions):
        layer_1 = f.relu(self.fc1(states))
        layer_1_cat = torch.cat([layer_1, actions], dim=1)
        layer_2 = f.relu(self.fc2(layer_1_cat))
        Q_value = self.fc3(layer_2)
        return Q_value