import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from utilities_func import *

class CustomFunction(nn.Module):
    """
    Torch Module Custom Function
    """
    def __init__(self, function, pars):
        """
        :param function: shape of function to assign to the module
        """
        super().__init__()
        self.function = function
        self.pars = pars

    def forward(self, x):
        """
        :param x: Input data
        :return: Output of custom activation function
        """
        x = torch.clamp(x,self.pars['lim_inf'],self.pars['lim_sup'])
        x = x.detach().numpy()
        return torch.tensor(self.function(x), dtype=torch.float32)


class MLP_Custom(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, genome, pars_w, pars):
        """
        :param genome: dna for genetic algorithm to create custom activation function
        :param pars_w:
        :param pars:
        """
        super().__init__()
        self.pars = pars
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.custom_activation = CustomFunction(self.act_func_generator(genome), self.pars)
        self.fc2 = nn.Linear(hidden_size, output_size)

        if pars_w is not None:
            self.set_weights(pars_w)
        else:
            self.init_weights()


        if self.pars['seed'] is not None:
            torch.manual_seed(self.pars['seed'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.custom_activation(x)
        x = self.fc2(x)
        return x

    def act_func_generator(self, genome):
        """
        :param genome: dna to interpolate
        :return: interpolated activation function
        """
        x_dom = np.linspace(self.pars['lim_inf'], self.pars['lim_sup'], len(genome))
        function = interp1d(x_dom, genome, kind='cubic', fill_value="extrapolate")
        return function

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def set_weights(self, pars_w):
        with torch.no_grad():
            for param, p in zip(self.parameters(), pars_w):
                param.copy_(torch.tensor(p))