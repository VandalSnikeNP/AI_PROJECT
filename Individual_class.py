from MLP_Custom_Class import *

class Individual(object):
    def __init__(self, genome, input_size, hidden_size, output_size, pars_w=None, pars=None):
        self.pars_w=pars_w
        self.pars = pars
        self.score = 0
        self.genome = genome
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.network = MLP_Custom(self.input_size, self.hidden_size, self.output_size, self.genome, self.pars_w, self.pars)
        self.init_pars_W = self.network.parameters()
        self.trained_pars_W = None

    def __getstate__(self):
        trained_params = [param.detach().numpy() for param in self.network.parameters()]
        init_params = [param.detach().numpy() for param in self.init_pars_W]

        state = {
            'genome': self.genome,
            'score': self.score,
            'trained_params': trained_params,
            'init_params': init_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'pars':self.pars,
            'pars_w':self.pars_w,
        }
        return state

    def __setstate__(self, state):
        self.genome = state['genome']
        self.score = state['score']
        self.input_size = state['input_size']
        self.hidden_size = state['hidden_size']
        self.output_size = state['output_size']
        self.pars = state['pars']
        self.pars_w = state['pars_w']
        self.network = MLP_Custom(self.input_size, self.hidden_size, self.output_size, self.genome, self.pars_w, self.pars)

        with torch.no_grad():
            for param, saved_param in zip(self.network.parameters(), state['trained_params']):
                param.copy_(torch.tensor(saved_param))

        self.init_pars_W = [torch.tensor(p) for p in state['init_params']]

    @classmethod
    def create_chromosome(self, level_inf, level_sup, step, dna_length, inter):
        """
        :param level_inf: lower limit of codomain
        :param level_sup: upper limit of domain
        :param step: granularity of codomain
        :param dna_length: length of dna (domain) sequence
        :param inter: integer values only
        :return: random dna following the characteristics
        """
        if not inter:
            dna = np.random.choice(np.arange(level_inf, level_sup, step), size=dna_length, replace=True)
        else:
            dna = np.random.randint(level_inf, level_sup, dna_length)
        return dna

    def training(self, train_loader, lr=0.001, epochs=1000, patience=5, early_stopping=True, device='cpu'):
        self.network.to(device)
        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_loss = float('inf')
        epochs_no_improve = 0
        training_losses = []

        for epoch in range(epochs):
            self.network.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = self.network(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            training_losses.append(avg_loss)

            if early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    # print(f'Early stopping triggered at Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
                    break
        # print(f'Ended at Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        self.score = training_losses[-1]
        self.trained_pars_W = self.network.parameters()

    @classmethod
    def selection(self, net_pop, set_k):
        subset = random.sample(net_pop, set_k)
        parent = min(subset, key=lambda x: x.score)
        return parent

    @staticmethod
    def mutation(genome,pars=None):

        method ='Flip_Bit_Mutation' if pars is None else pars['mut_method']
        hm_flip = 2 if pars is None else pars['hm_flip']
        hm_swap = 1 if pars is None else pars['hm_swap']

        if method == 'Flip_Bit_Mutation':
            for i in random.sample(range(len(genome)), hm_flip):
                genome[i] = round(np.random.choice(np.arange(pars['level_inf'], pars['level_sup'], pars['step']), size=1,replace=True)[0], 2)
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
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size, pars=self.pars)
            elif id == 'One_point_technique':
                child_chromosome = self.genome[:one_point] + second_parent.genome[one_point:]
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size, pars=self.pars)
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
                return Individual(child_genome, self.input_size, self.hidden_size, self.output_size, pars=self.pars)
        else:  # Crossover with mutation
            if id == 'Uniform_crossover':
                for x, y in zip(self.genome, second_parent.genome):
                    child_chromosome.append(round(random.choice([x, y]), 2))
                child_chromosome = Individual.mutation(child_chromosome)
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size, pars=self.pars)
            elif id == 'One_point_technique':
                child_chromosome = self.genome[:one_point] + second_parent.genome[one_point:]
                return Individual(child_chromosome, self.input_size, self.hidden_size, self.output_size, pars=self.pars)