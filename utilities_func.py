
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import numpy as np
import csv
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d
import pickle
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch

def dataset(pars,test_size=0.2,batch_size=10):
    X=np.linspace(pars['lim_inf'],pars['lim_sup'],pars['step_domain']).reshape(-1,1)
    Y=np.sin(X)
    X_tensor = torch.tensor(X,dtype=torch.float32)
    Y_tensor = torch.tensor(Y,dtype=torch.float32)
    X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=test_size, random_state=pars['seed'])
    train_data = TensorDataset(X_train, Y_train)
    val_data = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return X_train, X_val, Y_train, Y_val,train_loader, val_loader

def data_computing(net_pop,Mean_sim,Min_sim,scores_sim):
 scores = [individual.score for individual in net_pop]
 mean_score=sum(individual.score for individual in net_pop) / len(net_pop)
 min_score=min(net_pop,key=lambda x: x.score).score
 Mean_sim.append(round(mean_score, 9))
 Min_sim.append(round(min_score, 9))
 scores_sim.append(scores)
 return Mean_sim,Min_sim,scores

def print_par(index, pop):
    for name, param in pop[index].network.named_parameters():
        print(f'Ind:{index},{name}: {np.round(np.array(param.data), 3).reshape(1, -1)}')

def update_plots(Mean_sim, Min_sim, scores,net_data,pars):
    clear_output(wait=True)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))

    # Plot 1: Mean Simulation Trend
    ax1.plot(Mean_sim, marker='.')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Mean Simulation Trend: ' + str(round(Mean_sim[-1], 9)))

    if len(Mean_sim) >= 2:
        mean_legend_text = f'Prev: {Mean_sim[-2]}\nLast: {Mean_sim[-1]}'
    elif len(Mean_sim) == 1:
        mean_legend_text = f'Prev: N/A\nLast: {Mean_sim[-1]}'
    else:
        mean_legend_text = 'No data'
    ax1.legend([mean_legend_text], loc='upper right')

    # Plot 2: Min Simulation Trend
    ax2.plot(Min_sim, marker='.', color='red')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Min Score')
    ax2.set_title('Min Simulation Trend: ' + str(round(Min_sim[-1], 9)))

    if len(Min_sim) >= 2:
        min_legend_text = f'Prev: {Min_sim[-2]}\nLast: {Min_sim[-1]}'
    elif len(Min_sim) == 1:
        min_legend_text = f'Prev: N/A\nLast: {Min_sim[-1]}'
    else:
        min_legend_text = 'No data'
    ax2.legend([min_legend_text], loc='upper right')

    # Plot 3: Score Distribution
    ax3.hist(scores, bins=10, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Score Distribution in Current Generation')
    mean_score = np.mean(scores)
    ax3.axvline(mean_score, color='orange', linestyle='dashed', linewidth=1.5, label=f'Mean: {round(mean_score, 2)}')
    ax3.legend(loc='upper right')

    # Plot 4: Network Data Visualization
    x_doms = np.linspace(pars['lim_inf'], pars['lim_sup'], pars['step_domain'])

    # Generate activation function for net_data[-1]
    net_act_test_1 = act_func_generator(min(net_data[-1], key=lambda x: x.score).genome, pars['lim_inf'],
                                        pars['lim_sup'])
    y_doms_1 = net_act_test_1(x_doms)
    ax4.plot(x_doms, y_doms_1, label='Last Generation', color='green')

    # Generate activation function for net_data[-2], if available
    if len(net_data) > 1:
        net_act_test_2 = act_func_generator(min(net_data[-2], key=lambda x: x.score).genome, pars['lim_inf'],
                                            pars['lim_sup'])
        y_doms_2 = net_act_test_2(x_doms)
        ax4.plot(x_doms, y_doms_2, label='Previous Generation', color='red', linestyle='--')

    ax4.set_xlabel('Domain')
    ax4.set_ylabel('Activation')
    ax4.set_title('Network Activation Function')
    ax4.legend(loc='upper right')

    display(fig)
    plt.pause(0.01)
    plt.close(fig)

def save_data(formatted_time, net_data, pars):
    act_string = '_act_[' + str(pars['level_inf']) + ',' + str(pars['level_sup']) + ',' + str(pars['step']) + ',' + str(
        pars['lim_inf']) + ',' + str(pars['lim_sup']) + ',' + str(pars['step_domain']) + ',' + str(
        pars['dna_length']) + "]"

    training_string = "_train_[" + str(pars['input_size']) + ',' + str(pars['hidden_size']) + ',' + str(
        pars['output_size']) + ',' + str(pars['epochs']) + ',' + str(pars['lr']) + ',' + str(pars['seed']) + ']'

    gen_string = "_gen_[" + str(pars['pop_size']) + ',' + str(pars['N_gens']) + ',' + str(pars['set_k']) + ',' + str(
        pars['cross_method']) + ',' + str(pars['one_point']) + ',' + str(pars['length_per']) + ',' + str(
        pars['mut_method']) + ',' + str(pars['hm_flip']) + ',' + str(pars['hm_swap']) + ']'

    save_string = formatted_time + act_string + training_string + gen_string
    file_name = 'Results/' + save_string + '.pkl'

    if isinstance(net_data, (type((x for x in range(1))), type((lambda: (yield))()))):  # Generatore
        net_data = list(net_data)  # Converti il generatore in lista
    with open(file_name, 'wb') as f:
        pickle.dump(net_data, f)
        pickle.dump(pars, f)

    print("Data saved in Results/")

def save_parameters(formatted_time, net_data, pars):
    with open('Results/' + formatted_time + '_Population_parameters.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Parameter', 'Value'])
        for key, value in pars.items():
            writer.writerow([key, value])

        writer.writerow(['Generation', 'Individual', 'Genome', 'Score'])

        for gen_index, net_pop in enumerate(net_data):
            net_pop.sort(key=lambda x: x.score)
            for ind_index, ind in enumerate(net_pop):
                writer.writerow([gen_index, ind_index, ind.genome, ind.score])

    print('Data saved in ' + 'Results/' + formatted_time + '_Population_parameters.csv')

def save_score_data(formatted_time, net_data,pars):
    with open('Results/' + formatted_time + '_Generation_scores.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Parameter', 'Value'])
        for key, value in pars.items():
            writer.writerow([key, value])
        writer.writerow(['Gen', 'Mean Score', 'Min Score'])

        for gen_index, net_pop in enumerate(net_data):
            mean_score = round(sum(individual.score for individual in net_pop) / len(net_pop), 9)
            min_score = round(min(net_pop, key=lambda x: x.score).score, 9)
            writer.writerow([gen_index, mean_score, min_score])

    print('Data saved in ' + 'Results/' + formatted_time + '_Generation_scores.csv')

def load_data(file_name_loading):
    with open(file_name_loading, 'rb') as f:
        net_data = pickle.load(f)
        pars = pickle.load(f)
    return net_data,pars

def create_animation(Mean_sim, Min_sim, scores_sim, net_data, pars, N_gens, filename='simulation_animation.gif',
                     fps=10):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))
    anim = FuncAnimation(
        fig,
        update_plots_save,
        frames=N_gens,
        fargs=(Mean_sim, Min_sim, scores_sim, net_data, pars, fig, ax1, ax2, ax3, ax4),
        repeat=False
    )
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return anim

def update_plots_save(i, Mean_sim, Min_sim, scores_sim, net_data, pars, fig, ax1, ax2, ax3, ax4):
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()

    mean_sim = Mean_sim[:i + 1]
    min_sim = Min_sim[:i + 1]
    scores = scores_sim[i]

    fig.suptitle(f'Generation {i + 1}({i})/{pars["N_gens"]}', fontsize=16)

    # Figura 1: Mean Simulation Trend
    ax1.plot(mean_sim, marker='.')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean Score')
    ax1.set_title('Mean Simulation Trend: ' + str(round(mean_sim[-1], 9)))

    if len(mean_sim) >= 2:
        mean_legend_text = f'Prev: {mean_sim[-2]}\nLast: {mean_sim[-1]}'
    elif len(mean_sim) == 1:
        mean_legend_text = f'Prev: N/A\nLast: {mean_sim[-1]}'
    else:
        mean_legend_text = 'No data'
    ax1.legend([mean_legend_text], loc='upper right')

    # Figura 2: Min Simulation Trend
    ax2.plot(min_sim, marker='.', color='red')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Min Score')
    ax2.set_title('Min Simulation Trend: ' + str(round(min_sim[-1], 9)))

    if len(min_sim) >= 2:
        min_legend_text = f'Prev: {min_sim[-2]}\nLast: {min_sim[-1]}'
    elif len(min_sim) == 1:
        min_legend_text = f'Prev: N/A\nLast: {min_sim[-1]}'
    else:
        min_legend_text = 'No data'
    ax2.legend([min_legend_text], loc='upper right')

    # Figura 3: Score Distribution
    if i < len(scores_sim):
        ax3.hist(scores, bins=10, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distribution in Current Generation')
        mean_score = np.mean(scores)
        ax3.axvline(mean_score, color='orange', linestyle='dashed', linewidth=1.5,
                    label=f'Mean: {round(mean_score, 2)}')
        ax3.legend(loc='upper right')

    # Figura 4: Network Activation Functions
    x_doms = np.linspace(pars['lim_inf'], pars['lim_sup'], pars['step_domain'])
    if i < len(net_data):
        net_act_test_1 = act_func_generator(min(net_data[i], key=lambda x: x.score).genome, pars['lim_inf'],
                                            pars['lim_sup'])
        y_doms_1 = net_act_test_1(x_doms)
        ax4.plot(x_doms, y_doms_1, label=f'Generation {i}', color='green')

        if i > 0:
            net_act_test_2 = act_func_generator(min(net_data[i - 1], key=lambda x: x.score).genome, pars['lim_inf'],
                                                pars['lim_sup'])
            y_doms_2 = net_act_test_2(x_doms)
            ax4.plot(x_doms, y_doms_2, label=f'Generation {i - 1}', color='red', linestyle='--')

        ax4.set_xlabel('Domain')
        ax4.set_ylabel('Activation')
        ax4.set_title('Network Activation Function')
        ax4.legend(loc='upper right')

#Function to pass from DNA to activation function
def act_func_generator(DNA,dom_min,dom_max):
    x_dom=np.linspace(dom_min,dom_max,len(DNA))
    function=interp1d(x_dom,DNA,kind='cubic', fill_value="extrapolate")
    return function


def plot_some_activation_function(pars, net_pop, formatted_time, num_funcs=10):
    x_doms = np.linspace(pars['lim_inf'], pars['lim_sup'], pars['step_domain'])

    net_pop.sort(key=lambda x: x.score)
    selected = net_pop[:num_funcs]

    cols = 5
    rows = math.ceil(len(selected) / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    fig.suptitle('Best 10 individuals in the last generation', fontsize=16)

    for i, idx in enumerate(selected):
        function_test = act_func_generator(selected[i].genome, pars['lim_inf'], pars['lim_sup'])
        y_doms = function_test(x_doms)

        row = i // cols
        col = i % cols

        axs[row, col].plot(x_doms, y_doms)
        axs[row, col].set_title(f"Individual {i + 1}")
        axs[row, col].set_xlabel("x")
        axs[row, col].set_ylabel("y")
        min_score = min(net_pop, key=lambda x: x.score).score
        score_text = f"Score: {round(net_pop[i].score, 9)}"
        if net_pop[i].score == min_score:
            axs[row, col].text(0.5, -0.3, score_text, ha='center', va='top', transform=axs[row, col].transAxes,
                               color='g')
        else:
            axs[row, col].text(0.5, -0.3, score_text, ha='center', va='top', transform=axs[row, col].transAxes,
                               color='r')

    for j in range(pars['pop_size'], rows * cols):
        fig.delaxes(axs[j // cols, j % cols])

    plt.tight_layout()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig("Results/" + formatted_time + '_best_individuals.png', dpi=300, bbox_inches='tight')

    plt.show()
#------------------------------OTHERS------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print(torch.version.cuda)
# print(torch.cuda.is_available())

# Computing mean score and min score for a particular simulation
# Mean_sim=[round(sum(individual.score for individual in net_pop) / len(net_pop),9) for net_pop in net_data]
# Min_sim=[round(min(net_pop,key=lambda x: x.score).score,9) for net_pop in net_data]

#scores_sim = [[individual.score for individual in netp] for netp in net_data]
#display(f'Trained {i} and created pop {i+1} of : {len(new_pop)} individuals in {round(end_time - start_time,2)} seconds')