import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import math
import time
from datetime import datetime
from IPython.display import HTML
from utilities_func import *


# torch module for activation custom function
class CustomFunction(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        global pars
        x = torch.clamp(x, pars['lim_inf'], pars['lim_sup'])
        x = x.detach().numpy()
        return torch.tensor(self.function(x), dtype=torch.float32)


class MLP_Custom(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, genome, pars_W):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.custom_activation = CustomFunction(self.act_func_generator(genome))
        self.fc2 = nn.Linear(hidden_size, output_size)

        if pars_W is not None:
            self.set_weights(pars_W)
        else:
            self.init_weights()

        global pars
        if pars['seed'] is not None:
            torch.manual_seed(pars['seed'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.custom_activation(x)
        x = self.fc2(x)
        return x

    def act_func_generator(self, genome):
        global pars
        x_dom = np.linspace(pars['lim_inf'], pars['lim_sup'], len(genome))
        function = interp1d(x_dom, genome, kind='cubic', fill_value="extrapolate")
        return function

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def set_weights(self, pars_W):
        with torch.no_grad():
            for param, p in zip(self.parameters(), pars_W):
                param.copy_(torch.tensor(p))


class Individual(object):
    def __init__(self, genome, input_size, hidden_size, output_size, pars_W=None):
        self.genome = genome
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.score = 0
        self.network = MLP_Custom(self.input_size, self.hidden_size, self.output_size, self.genome, pars_W)
        self.init_pars_W = self.network.parameters()
        self.trained_pars_W = None

    # Create chromosome
    @classmethod
    def create_chromosome(self, level_inf, level_sup, step, dna_length, inter):
        if inter == False:
            dna = np.random.choice(np.arange(level_inf, level_sup, step), size=dna_length, replace=True)
        else:
            dna = np.random.randint(level_inf, level_sup, dna_length)
        return dna

    def training(self, lr, epochs, X, Y, patience=5):
        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr)

        loss_values_custom = []
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.network.train()
            optimizer.zero_grad()
            output = self.network(X)
            loss = mse_loss(output, Y)
            loss_values_custom.append(loss.item())
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                # print(f"Early stopping at epoch {epoch} due to no improvement.")
                break

        self.score = loss_values_custom[-1]
        self.trained_pars_W = self.network.parameters()

    @classmethod
    def selection(self, net_pop, set_k):
        subset = random.sample(net_pop, set_k)
        parent = min(subset, key=lambda x: x.score)
        return parent

    @staticmethod
    def mutation(genome, method='Flip Bit Mutation', hm_flip=2, hm_swap=1):
        global pars
        if method == 'Flip Bit Mutation':
            for i in random.sample(range(len(genome)), hm_flip):
                genome[i] = round(
                    np.random.choice(np.arange(pars['level_inf'], pars['level_sup'], pars['step']), size=1,
                                     replace=True)[0], 2)
            return genome
        else:
            for _ in range(hm_swap):
                i, j = random.sample(range(len(genome)), 2)
                genome[i], genome[j] = genome[j], genome[i]
                return genome

    # Crossover function
    def crossover(self, second_parent, id='Crossover_prof', one_point=2, length_per=10, mut=False):
        child_chromosome = []
        if mut is False:  # Crossover without mutation
            if id == 'Uniform_crossover':
                for x, y in zip(self.genome, second_parent.genome):
                    child_chromosome.append(round(random.choice([x, y]), 2))
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size)
            elif id == 'One_point_technique':
                child_chromosome = self.genome[:one_point] + second_parent.genome[one_point:]
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size)
            elif id == 'Crossover_prof':
                length = len(self.genome)
                p = int((length * length_per) / 100)
                end = length + 1
                start = 0
                while end > length:
                    start = np.random.randint(length)
                    end = start + p
                child_genome = second_parent.genome.copy()
                child_genome[start:end] = self.genome[start:end]
                return Individual(child_genome, self.input_size, self.hidden_size, self.output_size)
        else:  # Crossover with mutation
            if id == 'Uniform_crossover':
                for x, y in zip(self.genome, second_parent.genome):
                    child_chromosome.append(round(random.choice([x, y]), 2))
                child_chromosome = Individual.mutation(child_chromosome)
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size)
            elif id == 'One_point_technique':
                child_chromosome = self.genome[:one_point] + second_parent.genome[one_point:]
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size)


pars = {
    # Activation Parameters
    'level_inf': 0,     # Bottom level codomain activation function
    'level_sup': 3,     # Top level codomain activation function
    'inter': False,     # Select only integer values in codomain
    'step': 0.3,        # Granularity in the codomain
    'lim_inf': -7,      # Left limit of the domain activation function
    'lim_sup': 7,       # Right limit of the domain activation function
    'step_domain': 800, # Granularity in the dataset
    'dna_length': 28,   # Number of element in the domain

    # Training Parameters
    'input_size': 1,    # Input size neural network
    'hidden_size': 6,   # Hidden size neural network
    'output_size': 1,   # Output size neural network
    'epochs': 15,       # Number of epochs
    'lr': 0.01,         # Learning rate
    'seed': 18,         # Seed for weights generation

    # Genetic Algorithm Parameters
    'pop_size': 1000,       # Population size
    'N_gens': 50,           # Number of generation
    # Selection Parameters
    'set_k': 13,            # Size of the subset for selection
    # Crossover Parameters
    'cross_method':'Crossover_prof',  # 'Crossover_prof','Uniform_crossover','One_point_technique'
    'one_point':2,                    # Point where one_point_technique divide
    'length_per':10,                  # Percentage of the sub dna to change in the crossover_prof method
    # Mutuation Parameters
    'mut_method': 'Flip Bit Mutation',  # 'Flip Bit Mutation', 'others'
    'hm_flip': 2,                       # How many genes flip in the flip bit mutation
    'hm_swap': 2,                       # How many genes swap in the others method
}
# General parameters
pars['likely_per'] = int((pars['pop_size'] * 5) / 100)   # Percentage of the new population generated directly from the previous
pars['mut_per'] = int((pars['pop_size'] * 5) / 100)      # Percentage of the new population generated by mutation of the previous
pars['cross_per'] = int((pars['pop_size'] * 90) / 100)   # Percentage of the new population generated by crossover of the previous

