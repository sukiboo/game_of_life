import yaml
import pickle
import argparse
import numpy as np

import models
from game_of_life import GameOfLife


def test_algorithm(alg, params, train_params, exp_params, train_data, test_data):
    """Test an optimization algorithm on a model of a given type"""

    # parse experiment parameters
    dataset = exp_params['dataset']
    num_steps = exp_params['num_steps']
    num_tests = exp_params['num_tests']
    model_type = exp_params['model_type']
    activation = exp_params['activation']

    # train a model with a given algorithm
    logs = []
    print(f'\nTesting {alg} on {num_steps}-step GoL with {model_type} model'\
        + f' and {activation} activation on {dataset} dataset:\n')
    for t in range(num_tests):

        # create the model
        if model_type == 'recursive':
            model = models.create_model_recursive(num_steps, activation, random_seed=t)
        elif model_type == 'sequential':
            if num_steps == 1:
                # this case is covered by recursive model
                return
            model = models.create_model_sequential(num_steps, activation, random_seed=t)

        # train the model and save loss values
        model, history = models.train_model(model, alg, params,
                                            train_params, train_data, random_seed=t)
        logs.append([model.evaluate(*test_data), history])

    # save experiment logs
    exp_name = f'{num_steps}_{model_type}_{activation}_{alg}_{dataset}'
    print(f'{exp_name} success rate: {np.mean([l[0][1]==1 for l in logs]):.2f}\n')
    with open(f'./logs/{exp_name}_{num_tests}.pkl', 'wb') as logfile:
        pickle.dump(logs, logfile)


def search_parameters(algos, train_params, exp_params, train_data, test_data):
    """Perform hyperparameter search for all algorithms"""
    print(f'\n\n====== HYPERPARAMETER SEARCH {config_file} ======')
    for alg in ['Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'Adamax',
                'AdamW', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']:
        print(f'\n\ntesting {alg}\n')
        for name, params in algos.items():
            test_algorithm(name.replace('Algorithm', alg), params,
                           train_params, exp_params, train_data, test_data)


def run_experiments(config_file):
    """Setup and run experiments"""

    # read configs
    configs = yaml.safe_load(open(f'./configs/{config_file}'))
    exp_params = configs['exp_params']
    train_params = configs['train_params']
    algos = configs['algos']

    # parse experiment parameters
    dataset = exp_params['dataset']
    num_steps = exp_params['num_steps']
    num_tests = exp_params['num_tests']
    model_type = exp_params['model_type']
    activation = exp_params['activation']
    random_seed = exp_params['random_seed']

    # generate or load training set
    life = GameOfLife()
    if dataset == 'random':
        x_tr, y_tr = life.generate_dataset(
            steps=num_steps, num=train_params['epochs'],
            board_size=(64,64), density=.38, random_seed=random_seed)
    else:
        x_tr = np.expand_dims(np.load(f'./training_boards/{dataset}.npy'), [0,-1])
        y_tr = life.predict(x_tr, steps=num_steps)

    # generate evaluation test set
    x_ts, y_ts = life.generate_dataset(
        steps=num_steps, num=100, board_size=(100,100), density=.38, random_seed=random_seed)

    if config_file.startswith('search'):
        # perform hyperparameter search
        search_parameters(algos, train_params, exp_params, (x_tr,y_tr), (x_ts,y_ts))
    else:
        # run experiments
        for alg, params in algos.items():
            test_algorithm(alg, params, train_params, exp_params, (x_tr,y_tr), (x_ts,y_ts))


if __name__ == '__main__':

    # parse the configs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='1_tanh_fixed.yml',
                        help='name of the config file in "./configs/"')

    # read the inputs
    args = parser.parse_args()
    config_file = args.config

    # run experiments
    run_experiments(config_file)
