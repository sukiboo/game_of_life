"""
    remove this code before submission
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import os
from collections import defaultdict
from game_of_life import GameOfLife

sns.set_theme(style='darkgrid', palette='Paired', font='monospace', font_scale=1.)
##sns.set_theme(style='darkgrid', palette='muted', font='monospace', font_scale=1.)


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
    """Plot loss values"""
    fig, ax = plt.subplots(figsize=(8,5))
    for name, history in histories.items():
        ax.plot(history['loss'], linewidth=3, label=name)
        ##ax.plot(history['val_loss'], linewidth=3, linestyle='--')
    ##ax.set_yscale('log')
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
        ax.set_title(f'State at time {t}', size=24, weight='bold')
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


def visualize_success(logs_dir, save=False):
    """Plot convergence frequencies of different algorithms"""
    # compute convergence frequency
    conv = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for log_name in os.listdir(logs_dir):
        with open(f'./{logs_dir}/{log_name}','rb') as logfile:
            log_data = pickle.load(logfile)
        frequency = np.mean([l[0][1]==1 for l in log_data])
        steps, model_type, activation, algorithm, dataset, _ = log_name.split('_')
        conv[steps][model_type][activation][dataset][algorithm] = frequency
    # plot convergence frequency
    for steps in conv:
        for model_type in conv[steps]:
            for activation in conv[steps][model_type]:
                vals_random = conv[steps][model_type][activation]['random']
                vals_fixed = conv[steps][model_type][activation]['fixed']
                algs = sorted(vals_random.keys())
                fig, ax = plt.subplots(figsize=(8,4))
                for i in range(len(algs)):
                    ax.bar(i-.2, vals_random[algs[i]], width=.4)
                    ax.bar(i+.2, vals_fixed[algs[i]], width=.4)
                ax.set_title(f'Convergence rate on {model_type} model '\
                             + f'with {activation} activation after {steps} steps')
                plt.xticks(range(len(algs)), algs, rotation=30)
                plt.tight_layout()
                if save:
                    savename = f'convergence_{steps}_{model_type}_{activation}'
                    plt.savefig(f'./images/{savename}.png', dpi=300, format='png')
                plt.show()


def visualize_learning(logs_dir, save=False):
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
                    loss_fixed = Loss[steps][model_type][activation]['fixed'][algs[i]]
                    ax1.plot(loss_random, label='random', color=colors[(2*i)%12], linewidth=3)
                    ax1.plot(loss_fixed, label='fixed', color=colors[(2*i+1)%12], linewidth=3)
                    if (loss_random.size == 1) and (loss_fixed.size == 1):
                        plt.close()
                        continue

                    # plot average accuracy
                    acc_random = Acc[steps][model_type][activation]['random'][algs[i]]
                    acc_fixed = Acc[steps][model_type][activation]['fixed'][algs[i]]
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


def estimate_advantage(logs_dir):
    """Estimate how much faster algorithms converge on the fixed board"""
    # compute accuracy step
    T = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for log_name in os.listdir(logs_dir):
        with open(f'./{logs_dir}/{log_name}','rb') as logfile:
            log_data = pickle.load(logfile)

        # extract loss and accuracy
        inds = []
        for l in log_data:
            if l[0][1] == 1:
                # find first index where accuracy is 100%
                inds.append(np.where(np.array(l[1]['accuracy']) == 1)[0][0])
        steps, model_type, activation, algorithm, dataset, _ = log_name.split('_')
        T[steps][model_type][activation][dataset][algorithm] = np.mean(inds)

    # report average advantage for each case
    for steps in T:
        for model_type in T[steps]:
            for activation in T[steps][model_type]:
                algs = sorted(T[steps][model_type][activation]['random'].keys())
                for i in range(len(algs)):
                    t_random = T[steps][model_type][activation]['random'][algs[i]]
                    t_fixed = T[steps][model_type][activation]['fixed'][algs[i]]
                    if np.isnan(t_fixed) or np.isnan(t_random):
                        continue
                    advantage = 1 - t_fixed / t_random
                    print(f'{algs[i]} on {steps}-step {model_type} model with '\
                        + f'{activation} activation is {100*advantage:.2f}% faster')


def plot_logs(logs_dir, save=False):
    """Plot loss values from the logs in the given directory"""
    for log_name in os.listdir(logs_dir):
        try:
            with open(f'./{logs_dir}/{log_name}','rb') as logfile:
                log_data = pickle.load(logfile)
            fig, ax = plt.subplots(figsize=(8,5))
            for history in log_data:
                ax.plot(history[1]['loss'], linewidth=3)
            ax.set_ylim(-.01, .26)
            ax.set_title(log_name)
            plt.tight_layout()
            if save:
                plt.savefig(f'./images/{log_name[:-4]}.png', dpi=300, format='png')
            plt.show()
        except:
            pass


if __name__ == '__main__':

    ##plot_logs('./logs/test', save=False)
    ##visualize_density(1, save=True)
    ##visualize_success('./logs/100', save=False)
    visualize_learning('./logs/100', save=True)
    ##estimate_advantage('./logs/10')
    ##plot_glider(steps=4, save=True)
