import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
import torch.nn.init as init
import matplotlib.pyplot as plt
import math
import csv


class CustomFunction(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        global lim_inf, lim_sup
        x = torch.clamp(x, lim_inf, lim_sup)
        x_np = x.detach().cpu().numpy()  # Aggiungiamo .detach() per evitare l'errore
        output_np = self.function(x_np)
        return torch.tensor(output_np, dtype=torch.float32, device=x.device)


# %% Check device
device = torch.device("cuda")
print(f"Using device: {device}")
print(torch.version.cuda)
print(torch.cuda.is_available())


# %% Class MLP_CUSTOM
class MLP_Custom(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, genome, pars):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).to(device)
        self.custom_activation = CustomFunction(self.act_func_generator(genome)).to(device)
        self.fc2 = nn.Linear(hidden_size, output_size).to(device)

        if pars is not None:
            self.set_weights(pars)
        else:
            self.init_weights()

        global seed
        if seed is not None:
            torch.manual_seed(seed)

    def forward(self, x):
        x = self.fc1(x.to(device))
        x = self.custom_activation(x)
        x = self.fc2(x)
        return x

    def act_func_generator(self, genome):
        global lim_inf, lim_sup
        x_dom = np.linspace(lim_inf, lim_sup, len(genome))
        function = interp1d(x_dom, genome, kind='cubic', fill_value="extrapolate")
        return function

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)

    def set_weights(self, pars):
        with torch.no_grad():
            for param, p in zip(self.parameters(), pars):
                param.copy_(torch.tensor(p).to(device))


# %% Function to print parameters
def print_par(index, pop):
    for name, param in pop[index].network.named_parameters():
        print(f'Ind:{index}, {name}: {np.round(np.array(param.data.cpu()), 3).reshape(1, -1)}')


# %% Class Individual
class Individual(object):
    def __init__(self, genome, input_size, hidden_size, output_size, pars=None):
        self.genome = genome
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.score = 0
        self.network = MLP_Custom(self.input_size, self.hidden_size, self.output_size, self.genome, pars)
        self.init_pars = self.network.parameters()
        self.trained_pars = None

    @classmethod
    def create_chromosome(cls, level_inf, level_sup, step, dna_length, inter):
        if not inter:
            dna = np.random.choice(np.arange(level_inf, level_sup, step), size=dna_length, replace=True)
        else:
            dna = np.random.randint(level_inf, level_sup, dna_length)
        return dna

    def training(self, lr, epochs, X, Y):
        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr)

        loss_values_custom = []
        for epoch in range(epochs):
            self.network.train()
            optimizer.zero_grad()
            output = self.network(X)
            loss = mse_loss(output, Y)
            loss_values_custom.append(loss.item())
            loss.backward()
            optimizer.step()
        self.score = loss_values_custom[-1]
        self.trained_pars = self.network.parameters()

    @classmethod
    def selection(cls, net_pop, k):
        subset = random.sample(net_pop, k)
        parent = min(subset, key=lambda x: x.score)
        return parent

    @staticmethod
    def mutation(genome, method='Flip Bit Mutation', hm_flip=3, hm_swap=1):
        global level_inf, level_sup, step
        if method == 'Flip Bit Mutation':
            print(f'Ori gen:{genome}')
            for i in random.sample(range(len(genome)), hm_flip):
                genome[i] = round(np.random.choice(np.arange(level_inf, level_sup, step), size=1, replace=True)[0], 2)
            print(f'Mut gen:{genome}')
            return genome
        else:
            for _ in range(hm_swap):
                i, j = random.sample(range(len(genome)), 2)
                genome[i], genome[j] = genome[j], genome[i]
            return genome

    def crossover(self, second_parent, id=3, point=2, mut=False):
        child_chromosome = []
        if not mut:
            if id == 1:
                for x, y in zip(self.genome, second_parent.genome):
                    child_chromosome.append(round(random.choice([x, y]), 2))
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size)
            elif id == 2:
                child_chromosome = self.genome[:point] + second_parent.genome[point:]
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size)
            elif id == 3:
                length = len(self.genome)
                p = int((length * 20) / 100)
                end = length + 1
                start = 0
                while end > length:
                    start = np.random.randint(length)
                    end = start + p
                child_genome = second_parent.genome.copy()
                child_genome[start:end] = self.genome[start:end]
                return Individual(child_genome, self.input_size, self.hidden_size, self.output_size)
        else:
            if id == 1:
                for x, y in zip(self.genome, second_parent.genome):
                    child_chromosome.append(round(random.choice([x, y]), 2))
                child_chromosome = Individual.mutation(child_chromosome)
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size)
            elif id == 2:
                child_chromosome = self.genome[:point] + second_parent.genome[point:]
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size)


# %% Activation Parameters
level_inf = 0
level_sup = 3
inter = False
step = 0.2
lim_inf = -5
lim_sup = 5
dna_length = 10
input_size = 1
hidden_size = 8
output_size = 1
epochs = 200
lr = 0.01
seed = 42
pop_size =600
N_gens = 40
k = 3
likely_per = int((pop_size * 5) / 100)
mut_per = int((pop_size * 5) / 100)
cross_per = int((pop_size * 90) / 100)

# %% Initialization
net_pop = []
for i in range(pop_size):
    genome = Individual.create_chromosome(level_inf, level_sup, step, dna_length, inter)
    net_pop.append(Individual(genome, input_size, hidden_size, output_size))

# %% Data preparation
X = np.linspace(lim_inf, lim_sup, 300).reshape(-1, 1)
Y = np.sin(X)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

# %% Training Loop
net_data = [net_pop]
for i in range(N_gens):
    new_pop = []

    print(f'Training generation {i} started')
    for j in range(pop_size):
        net_pop[j].training(lr, epochs, X_tensor, Y_tensor)
    print(f'Training generation {i} ended')
    print_par(0, net_pop)

    for p in range(likely_per):
        new_pop.append(Individual.selection(net_pop, k))

    for p in range(mut_per):
        mut_chromosome = Individual.mutation(Individual.selection(net_pop, k).genome, method='Flip Bit Mutation')
        new_pop.append(Individual(mut_chromosome, input_size, hidden_size, output_size))

    for p in range(cross_per):
        first_parent = Individual.selection(net_pop, k)
        second_parent = Individual.selection(net_pop, k)
        child = first_parent.crossover(second_parent, 3, 2)
        new_pop.append(child)
    print(f'#:{len(net_pop)}')
    net_data.append(net_pop.copy())
    net_pop = new_pop
