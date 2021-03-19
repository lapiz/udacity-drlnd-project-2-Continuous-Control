import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers): #,  fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list) : number of nodes of hiddne layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layers = []

        input_size = state_size
        for layer in hidden_layers:
            self.layers.append( nn.Linear( input_size, layer) )
            input_size = layer
        
        self.layers.append( nn.Linear( input_size, action_size ) )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        return F.tanh(self.layers[-1](x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, func, hidden_layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            func (function) : activate function
            hidden_layers (list) : number of nodes of hiddne layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        if func == 'relu':
            self.func = F.relu
        elif func == 'leaky_relu':
            self.func = F.leaky_relu 

        self.layers = []

        self.layers.append( nn.Linear( state_size, hidden_layers[0]))

        input_size = hidden_layers[0] + action_size
        for layer in hidden_layers[1:]:
            self.layers.append( nn.Linear( input_size, layer) )
            input_size = layer            

        self.layers.append( nn.Linear(input_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.func(self.layers[0])
        x = torch.cat((xs, action), dim=1)

        for layer in self.layers[1:-1]:
            x = self.func(later(x))
        return self.layers[-1](x)
