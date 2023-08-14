"""
    This code creates all the figures presented in the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import time
import pickle
import os
from collections import defaultdict
from game_of_life import GameOfLife

sns.set_theme(style='darkgrid', palette='Paired', font='monospace', font_scale=1.)


def print_model_weights(model):
    """Display network architecture and weights"""
    print(f'\n\n{model.name} model weights:')
    for layer in model.layers[1:]:
        print('kernel:')
        try:
            print(np.squeeze(layer.get_weights()[0]).transpose(2,0,1))
        except:
            print(np.squeeze(layer.get_weights()[0]))
        print('bias:', layer.get_weights()[1], end='\n\n')


def plot_history(histories):
    """Plot loss values."""
    fig, ax = plt.subplots(figsize=(8,5))
    for name, history in histories.items():
        ax.plot(history['loss'], linewidth=3, label=name)
    ax.set_ylim(-.01, .26)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'./images/{int(time.time())}.png', dpi=300, format='png')
    plt.show()


def plot_glider(steps=1, save=False):
    """Plot glider for several steps"""
    life = GameOfLife()
    # set up initial state
    board = np.zeros((6,6))
    for i,j in [(1,1), (2,2), (2,3), (3,1), (3,2)]:
        board[i,j] = 1
    for t in range(steps+1):
        # plot the current state
        fig, ax = plt.subplots(figsize=(4,4))
        plt.pcolormesh(board, edgecolors='gray', linewidth=2)
        ax.axis('off')
        ##ax.set_title(f'State at time {t}', size=24, weight='bold')
        plt.tight_layout()
        if save:
            plt.savefig(f'./images/gol_state_{t}.png', dpi=300, format='png')
        plt.show()
        # predict the next state
        board = life.predict(np.expand_dims(board, [0,-1])).squeeze()


def visualize_density(steps, save=False):
    """Plot the average board density change after n steps"""
    # compute board density
    life = GameOfLife()
    density_in = np.linspace(0, 1, 101)
    density_out = []
    for density in density_in:
        x, y = life.generate_dataset(steps=steps, density=density,
            num=100, board_size=(100,100), random_seed=2023)
        density_out.append(y.mean())
    # plot board density
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(density_in, density_out, linewidth=3)
    ax.set_title(f'Board density after {steps} steps')
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., .5)
    plt.tight_layout()
    if save:
        plt.savefig(f'./images/density_{steps}.png', dpi=300, format='png')
    plt.show()


def visualize_density_combined(steps, save=False):
    """Plot the average board density change after throughout steps"""
    # compute board density
    life = GameOfLife()
    density = {t: [] for t in range(steps+1)}
    for d in np.linspace(0, 1, 101):
        x, _ = life.generate_dataset(density=d, num=100, board_size=(100,100))
        # compute average densities
        for t in range(steps+1):
            density[t].append(x.mean())
            x = life.predict(x)
    # plot board density
    sns.set_palette('tab10')
    fig, ax = plt.subplots(figsize=(5,3))
    for t in range(steps):
        ax.plot(density[0], density[t+1], linewidth=3, label=f'{t+1} steps')
    ax.set_xlim(0., 1.)
    ax.set_ylim(-.01, .4)
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'./images/gol_board_density.png', dpi=300, format='png')
    plt.show()


def visualize_success_datasets(logs_dir, save=False):
    """Plot convergence frequencies of different algorithms across multiple datasets"""
    # compute convergence frequency
    conv = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for log_name in os.listdir(logs_dir):
        with open(f'./{logs_dir}/{log_name}','rb') as logfile:
            log_data = pickle.load(logfile)
        frequency = np.mean([l[0][1]==1 for l in log_data])
        steps, model_type, activation, algorithm, dataset, _ = log_name.split('_')
        conv[steps][model_type][activation][dataset][algorithm] = frequency
    # plot convergence frequency
    print(conv)
    for steps in conv:
        for model_type in conv[steps]:
            for activation in conv[steps][model_type]:
                datasets = sorted(list(conv[steps][model_type][activation].keys()))
                sns.set_palette('Set2', n_colors=len(datasets))
                fig, ax = plt.subplots(figsize=(8,4))
                width = .8 / len(datasets)
                bars = [None] * len(datasets)
                for ind, dataset in enumerate(datasets):
                    vals = conv[steps][model_type][activation][dataset]
                    algs = sorted(vals.keys())
                    for i in range(len(algs)):
                        bars[ind] = ax.bar(i + (ind-.5)*width, vals[algs[i]],\
                                           width=width, color=sns.color_palette()[ind])
                ax.legend(bars, datasets)
                ax.set_title(f'Convergence rate on {model_type} model '\
                             + f'with {activation} activation after {steps} steps')
                plt.xticks(range(len(algs)), algs, rotation=30)
                plt.tight_layout()
                if save:
                    savename = f'success_{steps}_{model_type}_{activation}'
                    plt.savefig(f'./images/{savename}.png', dpi=300, format='png')
                plt.show()


def visualize_learning(logs_dir, fixed='fixed', save=False):
    """Plot losses accuracy trajectories of different algorithms"""
    # compute average losses and accuracies
    Loss = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    Acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for log_name in os.listdir(logs_dir):
        with open(f'./{logs_dir}/{log_name}','rb') as logfile:
            log_data = pickle.load(logfile)

        # extract loss and accuracy
        loss = np.array([l[1]['loss'] for l in log_data if l[0][1]==1]).mean(axis=0)
        acc = np.array([l[1]['accuracy'] for l in log_data if l[0][1]==1]).mean(axis=0)
        steps, model_type, activation, algorithm, dataset, _ = log_name.split('_')
        Loss[steps][model_type][activation][dataset][algorithm] = loss
        Acc[steps][model_type][activation][dataset][algorithm] = acc

    # plot losses and accuracies
    for steps in Loss:
        for model_type in Loss[steps]:
            for activation in Loss[steps][model_type]:
                algs = sorted(Loss[steps][model_type][activation]['random'].keys())
                colors = sns.color_palette()
                for i in range(len(algs)):
                    fig, ax1 = plt.subplots(figsize=(8,4))
                    ax2 = ax1.twinx()

                    # plot average loss
                    loss_random = Loss[steps][model_type][activation]['random'][algs[i]]
                    loss_fixed = Loss[steps][model_type][activation][fixed][algs[i]]
                    ax1.plot(loss_random, label='random', color=colors[(2*i)%12], linewidth=3)
                    ax1.plot(loss_fixed, label='fixed', color=colors[(2*i+1)%12], linewidth=3)
                    if (loss_random.size == 1) and (loss_fixed.size == 1):
                        plt.close()
                        continue

                    # plot average accuracy
                    acc_random = Acc[steps][model_type][activation]['random'][algs[i]]
                    acc_fixed = Acc[steps][model_type][activation][fixed][algs[i]]
                    ax2.plot(acc_random, label='random', color=colors[(2*i)%12], linewidth=3)
                    ax2.plot(acc_fixed, label='fixed', color=colors[(2*i+1)%12], linewidth=3)

                    ax1.set_title(f'{algs[i]} on {model_type} model '\
                                + f'with {activation} activation after {steps} steps')
                    ##ax1.legend()
                    ax2.grid(False)
                    plt.tight_layout()
                    if save:
                        savename = f'losses_{steps}_{model_type}_{activation}_{algs[i]}'
                        plt.savefig(f'./images/{savename}.png', dpi=300, format='png')
                        plt.close()
                    else:
                        plt.show()


def estimate_advantage(logs_dir, fixed='fixed'):
    """Estimate how much faster algorithms converge on the fixed board"""
    T = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    S = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for log_name in sorted(os.listdir(logs_dir)):
        with open(f'./{logs_dir}/{log_name}','rb') as logfile:
            log_data = pickle.load(logfile)
        print(log_name)
        # extract loss and accuracy
        inds = []
        for l in log_data:
            ind = len(l[1]['accuracy']) - 1
            while (l[1]['accuracy'][ind] == 1) and (ind > 0):
                ind -= 1
            if ind < len(l[1]['accuracy']) - 1:
                inds.append(ind)

        steps, model_type, activation, algorithm, dataset, _ = log_name.split('_')
        T[steps][model_type][activation][dataset][algorithm] = np.mean(inds)
        S[steps][model_type][activation][dataset][algorithm] = len(inds) / len(log_data)

    print(T)
    print(S)

    # report average advantage for each case
    algs = ['Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'AdamW', \
            'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']
    for steps in T:
        for model_type in T[steps]:
            for activation in T[steps][model_type]:
                print(f'\n{steps}-step {model_type} model with {activation} activation:\n')

                # setup latex table
                table_setup = [
                    r'\begin{tabular}{lrrrrrr}',
                    r'\toprule',
                    r'& \multicolumn{3}{c}{Success rate} & \multicolumn{3}{c}{Number of epochs}',
                    r'\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
                    r'algorithm & random & fixed & change & random & fixed & change',
                    r'\\\midrule']
                print(*table_setup, sep='\n')

                # add metrics for each algorithm
                for i in range(len(algs)):

                    # extract success rate
                    s_random = S[steps][model_type][activation]['random'][algs[i]]
                    s_fixed = S[steps][model_type][activation][fixed][algs[i]]
                    print(f'{algs[i]} & {s_random:.2f} & {s_fixed:.2f}'.replace('0.00', '---'), end='')
                    if s_random == 0 or s_fixed == 0:
                        print(f' & ---', end='')
                    else:
                        s_advantage = s_fixed / s_random - 1
                        print(f' & {100*s_advantage:+.0f}\%', end='')

                    # extract average number of iterations
                    t_random = T[steps][model_type][activation]['random'][algs[i]]
                    t_fixed = T[steps][model_type][activation][fixed][algs[i]]
                    print(f' & {t_random:.0f} & {t_fixed:.0f}'.replace('nan', '---'), end='')
                    t_advantage = 1 - t_fixed / t_random
                    if np.isnan(t_advantage):
                        print(f' & ---', end='')
                    else:
                        print(f' & {100*t_advantage:+.0f}\%', end='')
                    print('\n' + r'\\')

                # close latex table
                print(r'\bottomrule', r'\end{tabular}', sep='\n')


def plot_board(dataset, steps=0, name=None, save=False):
    """Plot the state of GoL board throughout the steps"""
    life = GameOfLife()
    if dataset == 'random':
        board, _ = life.generate_dataset(density=.38,\
            num=1, board_size=(64,64), random_seed=2023)
        board = board.squeeze()
    else:
        board = np.load(f'./training_boards/{dataset}.npy')
    for t in range(steps+1):
        # plot the current state
        fig, ax = plt.subplots(figsize=(8,8))
        plt.pcolormesh(board, edgecolors='gray', linewidth=.5)
        ax.axis('off')
        plt.tight_layout()
        if save:
            name = 'random' if dataset=='random' else 'fixed'
            plt.savefig(f'./images/gol_board_{name}_{t}.png', dpi=300, format='png')
        plt.show()
        # predict the next state
        board = life.predict(np.expand_dims(board, [0,-1])).squeeze()


def plot_patterns(save=False):
    """Plot the patterns of GoL"""
    patterns = [
        [[1,0,1], [0,1,0], [1,0,1]],
        [[0,1,0], [1,0,1], [0,1,0]],
        [[0,1,1], [1,0,1], [1,1,0]],
        [[1,0,1], [0,0,0], [0,1,0]],

        [[1,0,1], [0,0,0], [1,0,1]],
        [[1,0,1], [1,1,0], [1,0,1]],
        [[1,1,1], [1,0,1], [1,1,1]],
        [[1,0,1], [0,1,0], [0,1,0]],
    ]

    # plot each pattern
    for i, pattern in enumerate(patterns):
        fig, ax = plt.subplots(figsize=(2,2))
        plt.pcolormesh(pattern, edgecolors='gray', linewidth=1)
        ax.axis('off')
        plt.tight_layout()
        if save:
            plt.savefig(f'./images/gol_pattern_{i}.png', dpi=300, format='png')
        plt.show()


def display_hyperparameters(logs_dir):
    """Display success rates for each alogrithm and parameter"""
    params = ['1e-1', '3e-2', '1e-2', '3e-3', '1e-3', '3e-4', '1e-4']
    print(f'\nLoading {logs_dir}...')
    for algo in sorted(os.listdir(logs_dir)):
        print(f'\n{algo} hyperparameters success rate:')
        for i, log_name in enumerate(sorted(os.listdir(logs_dir + algo))):
            with open(logs_dir + algo + '/' + log_name,'rb') as logfile:
                log_data = pickle.load(logfile)
            print(f'{params[i]} -- {np.mean([l[0][1]==1 for l in log_data]):.2f}')


def parse_log_data(logs_dir):
    """Load experiment data from log files."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for env_dir in sorted(os.listdir(logs_dir)):
        for dataset in sorted(os.listdir('/'.join([logs_dir, env_dir]))):
            for activation in sorted(os.listdir('/'.join([logs_dir, env_dir, dataset]))):
                for alg in sorted(os.listdir('/'.join([logs_dir, env_dir, dataset, activation]))):
                    alg_dir = '/'.join([logs_dir, env_dir, dataset, activation, alg, ''])
                    for i, log_name in enumerate(sorted(os.listdir(alg_dir))):
                        with open(alg_dir + log_name, 'rb') as logfile:
                            log_data = pickle.load(logfile)
                        data[env_dir][dataset][activation][alg]\
                            .append(np.mean([l[0][1]==1 for l in log_data]))
    print(data)
    return data


def plot_search(logs_dir, save=False):
    """Visualize convergence rates for each alogrithm and parameter."""
    data = parse_log_data(logs_dir)
    params = ['1e-1', '3e-2', '1e-2', '3e-3', '1e-3', '3e-4', '1e-4']
    for env in data:
        for dataset in data[env]:
            for activation in data[env][dataset]:
                for alg in data[env][dataset][activation]:
                    color_relu = 'royalblue' if dataset=='random' else 'salmon'
                    color_tanh = 'lightskyblue' if dataset=='random' else 'peachpuff'
                    fig, ax = plt.subplots(figsize=(5,2))
                    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x: .2f}'))
                    ax.bar(np.arange(len(params))-.2, data[env][dataset]['relu'][alg],
                                     width=.4, color=color_relu)
                    ax.bar(np.arange(len(params))+.2, data[env][dataset]['tanh'][alg],
                                     width=.4, color=color_tanh)
                    plt.xticks(range(len(params)), params, rotation=0)
                    plt.legend(['relu', 'tanh'])
                    plt.tight_layout()
                    if save:
                        savename = f'search_{env}_{dataset}_{alg.lower()}'
                        plt.savefig(f'./images/search/{savename}.png', dpi=300, format='png')
                    else:
                        plt.show()


def visualize_success(logs_dir, fixed='fixed', save=False):
    """Plot convergence frequencies of different algorithms"""
    sns.set_palette('muted', n_colors=2)
    # compute convergence frequency
    conv = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for log_name in os.listdir(logs_dir):
        with open(f'./{logs_dir}/{log_name}','rb') as logfile:
            log_data = pickle.load(logfile)
        frequency = np.mean([l[0][1]==1 for l in log_data])
        steps, model_type, activation, algorithm, dataset, _ = log_name.split('_')
        conv[steps][model_type][activation][dataset][algorithm] = frequency
    # plot convergence frequency
    print(conv)
    for steps in conv:
        for model_type in conv[steps]:
            for activation in conv[steps][model_type]:
                vals_random = conv[steps][model_type][activation]['random']
                vals_fixed = conv[steps][model_type][activation][fixed]
                algs = sorted(vals_random.keys())
                fig, ax = plt.subplots(figsize=(8,4))
                for i in range(len(algs)):
                    bar_random = ax.bar(i-.2, vals_random[algs[i]], width=.4)
                    bar_fixed = ax.bar(i+.2, vals_fixed[algs[i]], width=.4)
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x: .2f}'))
                plt.xticks(np.arange(len(algs))+.25, algs, rotation=30, ha='right')
                ax.legend([bar_random, bar_fixed], ['random', 'fixed'], loc='upper right')
                plt.tight_layout()
                if save:
                    savename = f'success_{steps}_{model_type}_{activation}'
                    plt.savefig(f'./images/{savename}.png', dpi=300, format='png')
                plt.show()



if __name__ == '__main__':

    plot_board('fixed', steps=2, save=True)
    ##display_hyperparameters('./logs/search/1_step_rec/fixed/tanh/')
    ##plot_search('./logs/search', save=True)
    ##plot_patterns(save=True)
    ##plot_glider(steps=5, save=True)
    ##visualize_success('./logs/final/', save=True)
    ##estimate_advantage('./logs/final')

    ##visualize_density_combined(steps=5, save=True)
    ##visualize_success_datasets('./logs/1_step_results', save=False)
