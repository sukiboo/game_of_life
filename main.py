import numpy as np
from game_of_life import GameOfLife
import models
import visualization as viz


if __name__ == '__main__':

    # number of steps in GoF
    steps = 2
    epochs = 10000

    # ground truth model
    model = models.get_ground_truth_model(steps)
    print(model.summary())

    # generate train and test data
    life = GameOfLife()
    x_tr, y_tr = life.generate_dataset(
        steps=steps, num=100, board_size=(32,32), density=.25, random_seed=0)
    x_ts, y_ts = life.generate_dataset(
        steps=steps, num=100, board_size=(32,32), density=.25, random_seed=1)
    data = ((x_tr,y_tr), (x_ts,y_ts))


    # train model via backpropagation
    params_bp = {'name': 'Adam', 'learning_rate': 1e-3}
    model_bp, history_bp = models.train_model_bp(steps, data, epochs, params_bp)

    # train model via genetic algorithm
    params_ga = {'population': 100, 'parents': 10, 'sigma': .01}
    model_ga, history_ga = models.train_model_ga(steps, data, epochs, params_ga)

    # train model via smoothing-based optimization
    params_so = {'sigma': .01, 'learning_rate': 1e-2, 'num_quad': 7}
    model_so, history_so = models.train_model_so(steps, data, epochs, params_so)

    # train model via evolution strategy
    params_es = {'population': 100, 'sigma': .01, 'learning_rate': 1e-3}
    model_es, history_es = models.train_model_es(steps, data, epochs, params_es)

    # train model via particle swarm optimization
    params_ps = {'num_particles': 100}
    model_ps, history_ps = models.train_model_ps(steps, data, epochs, params_ps)

    # visualize model training
    viz.plot_history({'model_bp': history_bp, 'model_ga': history_ga,
        'model_so': history_so, 'model_es': history_es, 'model_ps': model_ps})

