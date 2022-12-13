
import numpy as np
import tensorflow as tf

from game_of_life import GameOfLife
import visualization as viz

np.set_printoptions(precision=3, suppress=True)
tf.random.set_seed(0)


def create_model(n=1):
    '''construct feedforward convolutional network to learn n-step GoF'''
    model = tf.keras.models.Sequential(n * [
            tf.keras.layers.Conv2D(2, kernel_size=(3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation='relu'),
            ])
    return model


if __name__ == '__main__':

    # number of steps in GoF
    n = 2

    # generate train and test data
    life = GameOfLife()
    x_tr, y_tr = life.generate_dataset(step=n, num=10000, board_size=(32,32), random_seed=0)
    x_ts, y_ts = life.generate_dataset(step=n, num=1000, board_size=(32,32), random_seed=1)

    # create model
    model = create_model()

    # compile, train, evaluate model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x_tr, y_tr, batch_size=100, epochs=20)
    model.evaluate(x_ts, y_ts)

    # show results
    viz.print_model_weights(model)

