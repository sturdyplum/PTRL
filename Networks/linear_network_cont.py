import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

"""
Creates a actor critic network comprised of only linear layers. 
    `layer_inputs`:
        the length of this will be number of layers. the ith 
        value will be the number of inputs that the ith layer
        will have. The length >= 2.
    
    `num_actions`: 
        the number of actions that the network should have. 
        this is the number of outputs that will be returned 
        by the policy.
"""

class LinearNetCont(nn.Module):
    
    def __init__(self, layer_inputs, action_space , activation_function, hidden_states):
        super(LinearNetCont, self).__init__()
        self.activation_function = activation_function
        self.action_space = action_space
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # There should at least be two layers to the network. 
        assert len(layer_inputs) > 1

        self.linears = []
        for i in range(len(layer_inputs) - 1):
            self.linears.append(nn.Linear(layer_inputs[i], layer_inputs[i+1]))
            nn.init.orthogonal_(self.linears[-1].weight, 1.0)
            self.linears[-1].bias.data.fill_(0.0)

        # If we don't wrap the list here the model wont know
        # these have trainable paramters.
        self.linears = nn.ModuleList(self.linears)

        self.gru = nn.GRU(layer_inputs[-1], hidden_states, 1)

        self.value = nn.Linear(layer_inputs[-1] + hidden_states, 1)
        nn.init.orthogonal_(self.value.weight, 1.0)
        self.value.bias.data.fill_(0.0)

        self.mu_policy = nn.Linear(layer_inputs[-1] + hidden_states, action_space.shape[0])
        nn.init.orthogonal_(self.mu_policy.weight, 1.0)
        self.mu_policy.bias.data.fill_(0.0)

        self.var_policy = nn.Linear(layer_inputs[-1] + hidden_states, action_space.shape[0])
        nn.init.orthogonal_(self.var_policy.weight, 1.0)
        self.var_policy.bias.data.fill_(0.0)
    
    def select_actions(self, dist):
        return torch.tensor(self.action_space.low) + (dist.sample() + 1)/2 * torch.tensor(self.action_space.high - self.action_space.low)

    def normalize_actions(self, actions):
        perc = (actions - torch.tensor(self.action_space.low)) / torch.tensor(self.action_space.high - self.action_space.low)
        perc *= 2
        perc -= 1
        return perc


    def forward(self, x, hidden_state):
        for linear in self.linears:
            x = self.activation_function(linear(x))
        
        y, new_hidden_state = self.gru(x.unsqueeze(0), hidden_state)
        y = self.activation_function(y.squeeze(0))
        x = torch.cat((x, y), 1)

        mu_output = self.tanh(self.mu_policy(x))
        var_output = self.tanh(self.var_policy(x)) + 1
        
        return Normal(mu_output, var_output), torch.flatten(self.value(x)), new_hidden_state
        