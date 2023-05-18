import numpy as np
from game_of_life import GameOfLife
import models
import visualization as viz
import pickle


def test_algorithm(params_opt, params_train, activation, model_type):
    """Test an optimization algorithm on a model of a given type"""
    logs = []
    for i in range(num_exp):

        # create model
        if model_type == 'recursive':
            model = models.create_model_recursive(steps, activation, random_seed=i)
        elif model_type == 'sequential':
            if steps == 1:
                # this case is covered by recursive model
                return
            model = models.create_model_sequential(steps, activation, random_seed=i)

        # train model
        if params_opt['name'].startswith('DGS'):
            model, history = models.train_model_so(model, (x_tr,y_tr), params_opt, params_train)
        elif params_opt['name'] == 'GA':
            model, history = models.train_model_ga(model, (x_tr,y_tr), params_opt, params_train)
        else:
            model, history = models.train_model_bp(model, (x_tr,y_tr), params_opt, params_train)

        # save loss values
        logs.append([model.evaluate(x_ts, y_ts), history])

    exp_name = f'{steps}_{model_type}_{activation}_{params_opt["name"]}_{dataset}'
    print(f'{exp_name} success rate: {np.mean([l[0][1]==1 for l in logs]):.2f}\n')
    with open(f'./logs/{exp_name}_{num_exp}.pkl', 'wb') as logfile:
        pickle.dump(logs, logfile)



if __name__ == '__main__':

    # number of steps in GoF
    steps = 2
    num_exp = 100

    ##"""
    life = GameOfLife()
    for steps in [1,2]:
        for dataset in ['random']:

            # generate evaluation test set
            x_ts, y_ts = life.generate_dataset(
                steps=steps, num=100, board_size=(100,100), density=.38, random_seed=2023)
            # get training set
            if dataset == 'fixed':
                x_tr = np.expand_dims(np.load('./training_boards/x9_tr.npy'), [0,-1])
                y_tr = life.predict(x_tr, steps=steps)
            else:
                x_tr, y_tr = life.generate_dataset(
                    steps=steps, num=30000, board_size=(32,32), density=.38, random_seed=2023)


            # parameters for optimization algorithms
            params_train = {'batch_size': 1, 'steps_per_epoch': 1, 'epochs': 30000}
            params = [
                {'name': 'Adadelta', 'learning_rate': 1e-1},
                {'name': 'Adafactor', 'learning_rate': 2e-2},#2.12
                {'name': 'Adagrad', 'learning_rate': 3e-2},
                {'name': 'Adam', 'learning_rate': 3e-4},
                {'name': 'AdamW', 'learning_rate': 3e-4},#2.12
                {'name': 'Adamax', 'learning_rate': 1e-3},
                {'name': 'Ftrl', 'learning_rate': 5e-2},
                {'name': 'Nadam', 'learning_rate': 3e-4},
                {'name': 'RMSprop', 'learning_rate': 3e-4},
                {'name': 'SGD', 'momentum': .5, 'learning_rate': 1e-2},
            ]

            # run experiments
            for params_opt in params:
                for activation in ['relu', 'tanh']:
                    for model_type in ['recursive', 'sequential']:
                        test_algorithm(params_opt, params_train, activation, model_type)
    ##"""

    """
    # generate evaluation test set
    life = GameOfLife()
    x_ts, y_ts = life.generate_dataset(
        steps=steps, num=100, board_size=(100,100), density=.38, random_seed=2023)

    # load training set
    dataset = 'fixed'
    x_tr = np.expand_dims(np.load('./training_boards/x9_tr.npy'), [0,-1])
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


