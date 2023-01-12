
from game_of_life import GameOfLife
import models
import visualization as viz


if __name__ == '__main__':

    # number of steps in GoF
    n = 2
    epochs = 100

    # generate train and test data
    life = GameOfLife()
    x_tr, y_tr = life.generate_dataset(step=n, num=10000, board_size=(32,32), random_seed=0)
    x_ts, y_ts = life.generate_dataset(step=n, num=1000, board_size=(32,32), random_seed=1)
    data = ((x_tr,y_tr), (x_ts,y_ts))

    # ground truth model
    model = models.get_ground_truth_model(n)
    print(model.summary())
    viz.print_model_weights(model)

    # train model via backpropagation
    params_bp = {'name': 'Adam', 'learning_rate': 1e-3}
    model_bp, history_bp = models.train_model_bp(n, data, epochs, params_bp)
    ##viz.print_model_weights(model_bp)

    # train model via genetic algorithm
    params_ga = {'population': 100, 'parents': 10, 'sigma': .1}
    model_ga, history_ga = models.train_model_ga(n, data, epochs, params_ga)
    ##viz.print_model_weights(model_ga)

    # train model via evolution strategy
    params_es = {'population': 100, 'sigma': .1, 'learning_rate': 1e-2}
    model_es, history_es = models.train_model_es(n, data, epochs, params_es)
    ##viz.print_model_weights(model_es)

    # train model via smoothing-based optimization
    params_so = {'sigma': .1, 'learning_rate': 1e-2, 'num_quad': 5}
    model_so, history_so = models.train_model_so(n, data, epochs, params_so)
    ##viz.print_model_weights(model_so)

    # visualize model training
    viz.plot_history({'model_bp': history_bp, 'model_ga': history_ga,
                      'model_es': history_es, 'model_so': history_so})

