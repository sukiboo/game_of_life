
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from game_of_life import GameOfLife

tf.random.set_seed(0)


if __name__ == '__main__':

    life = GameOfLife()

    x_tr, y_tr = life.generate_dataset(num=1000, board_size=(32,32), random_seed=0)
    x_ts, y_ts = life.generate_dataset(num=100, board_size=(32,32), random_seed=1)


    ##model = tf.keras.models.Sequential([
            ##tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'),
            ##tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'),
            ##tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', activation=None),
            ##])
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(2, kernel_size=(3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', activation=None),
            ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    model.fit(x_tr, y_tr, batch_size=8, epochs=10)
    model.evaluate(x_ts, y_ts)

    print(model(x_ts[:1]).numpy().squeeze(), y_ts[:1].squeeze(), sep='\n')
