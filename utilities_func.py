
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import numpy as np
import csv
from datetime import datetime

def print_par(index, pop):
    for name, param in pop[index].network.named_parameters():
        print(f'Ind:{index},{name}: {np.round(np.array(param.data), 3).reshape(1, -1)}')

def update_plots(Mean_sim, Min_sim, scores):
    clear_output(wait=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.cla()
    ax2.cla()

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

    ax3.hist(scores, bins=10, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Score Distribution in Current Generation')
    mean_score=np.mean(scores)
    ax3.axvline(mean_score, color='orange', linestyle='dashed', linewidth=1.5, label=f'Mean: {round(mean_score, 2)}')
    ax3.legend(loc='upper right')

    display(fig)
    plt.pause(0.01)
    plt.close(fig)



def save_parameters(formatted_time, net_data):
    with open('Results/Population_parameters_' + formatted_time + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'Individual', 'Genome', 'Score'])

        for gen_index, net_pop in enumerate(net_data):
            net_pop.sort(key=lambda x: x.score)
            for ind_index, ind in enumerate(net_pop):
                writer.writerow([gen_index, ind_index, ind.genome, ind.score])

    print('Data saved in ' + 'Results/Population_parameters_' + formatted_time + '.csv')


def save_score_data(formatted_time, net_data):
    with open('Results/Generation_scores_' + formatted_time + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Gen', 'Mean Score', 'Min Score'])

        for gen_index, net_pop in enumerate(net_data):
            mean_score = round(sum(individual.score for individual in net_pop) / len(net_pop), 9)
            min_score = round(min(net_pop, key=lambda x: x.score).score, 9)
            writer.writerow([gen_index, mean_score, min_score])

    print('Data saved in ' + 'Results/Generation_scores_' + formatted_time + '.csv')







#------------------------------OTHERS------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print(torch.version.cuda)
# print(torch.cuda.is_available())

# Computing mean score and min score for a particular simulation
# Mean_sim=[round(sum(individual.score for individual in net_pop) / len(net_pop),9) for net_pop in net_data]
# Min_sim=[round(min(net_pop,key=lambda x: x.score).score,9) for net_pop in net_data]