import numpy as np
from game_of_life import GameOfLife
import models
import visualization as viz
import pickle
import yaml


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
    for t in range(num_tests):

        # create the model
        if model_type == 'recursive':
            model = models.create_model_recursive(num_steps, activation, random_seed=t)
        elif model_type == 'sequential':
            if num_steps == 1:
                # this case is covered by recursive model
                return
            model = models.create_model_sequential(num_steps, activation, random_seed=t)

        # train the model
        ##model, history = models.train_model(model, alg, params, train_params, train_data)
        # hyperparameter search clunky workaround
        model, history = models.train_model(model, alg[:-1], params, train_params, train_data)

        # save loss values
        logs.append([model.evaluate(*test_data), history])

    exp_name = f'{num_steps}_{model_type}_{activation}_{alg}_{dataset}'
    print(f'{exp_name} success rate: {np.mean([l[0][1]==1 for l in logs]):.2f}\n')
    with open(f'./logs/{exp_name}_{num_tests}.pkl', 'wb') as logfile:
        pickle.dump(logs, logfile)


if __name__ == '__main__':

    config_file = 'search_1_adadelta.yml'

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
            board_size=(32,32), density=.38, random_seed=random_seed)
    else:
        x_tr = np.expand_dims(np.load(f'./training_boards/{dataset}.npy'), [0,-1])
        y_tr = life.predict(x_tr, steps=num_steps)

    # generate evaluation test set
    x_ts, y_ts = life.generate_dataset(
        steps=num_steps, num=100, board_size=(100,100), density=.38, random_seed=random_seed)

    # run experiments
    for alg, params in algos.items():
        test_algorithm(alg, params, train_params, exp_params, (x_tr,y_tr), (x_ts,y_ts))


    """
    # generate evaluation test set
    life = GameOfLife()
    x_ts, y_ts = life.generate_dataset(
        steps=steps, num=100, board_size=(100,100), density=.38, random_seed=2023)

    # load training set
    dataset = 'fixed'
    ##x_tr = np.expand_dims(np.load('./training_boards/x9_tr.npy'), [0,-1])
    x_tr = np.expand_dims(np.load('./training_boards/x_old_tr.npy'), [0,-1])
    y_tr = life.predict(x_tr, steps=steps)
    ### generate trining set
    ##dataset = 'random'
    ##x_tr, y_tr = life.generate_dataset(
        ##steps=steps, num=10000, board_size=(32,32), density=.38, random_seed=2023)

    # parameters for optimization algorithms
    params_train = {'batch_size': 1, 'steps_per_epoch': 1, 'epochs': 30000}
    params = [
        {'name': 'Adadelta', 'learning_rate': 1e-1},
        {'name': 'Adafactor', 'learning_rate': 2e-2},#?
        {'name': 'Adagrad', 'learning_rate': 3e-2},
        {'name': 'Adam', 'learning_rate': 3e-4},
        {'name': 'AdamW', 'learning_rate': 3e-4},#?
        {'name': 'Adamax', 'learning_rate': 1e-3},
        {'name': 'Ftrl', 'learning_rate': 5e-2},
        {'name': 'Nadam', 'learning_rate': 3e-4},
        {'name': 'RMSprop', 'learning_rate': 3e-4},
        {'name': 'SGD', 'momentum': .5, 'learning_rate': 1e-2},
    ]

    '''
    for params_opt in params:
        for activation in ['relu', 'tanh']:
            for model_type in ['recursive', 'sequential']:
                test_algorithm(params_opt, params_train, activation, model_type)
    '''

    ##params_opt = {'name': 'AdamW', 'learning_rate': 3e-4}
    ##test_algorithm(params_opt, params_train, activation='tanh', model_type='recursive')
    """


