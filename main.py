
import numpy as np
import tensorflow as tf

from game_of_life import GameOfLife
import visualization as viz

np.set_printoptions(precision=3, suppress=True)
tf.random.set_seed(0)


if __name__ == '__main__':

    # generate train and test data
    life = GameOfLife()
    x_tr, y_tr = life.generate_dataset(num=10000, board_size=(32,32), random_seed=0)
    x_ts, y_ts = life.generate_dataset(num=1000, board_size=(32,32), random_seed=1)

    # create model
    ### my model
    ##model = tf.keras.models.Sequential([
            ##tf.keras.layers.Conv2D(2, kernel_size=(3,3), padding='same', activation='relu'),
            ##tf.keras.layers.Conv2D(1, kernel_size=(3,3), padding='same', activation='relu'),
            ##])

    ### author's model with fixed last layer -- they use 20, -10 weights
    ##model = tf.keras.models.Sequential([
            ##tf.keras.layers.Conv2D(2, kernel_size=(3,3), padding='same', activation='relu'),
            ##tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation='relu'),
            ##tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation='sigmoid',
                ##trainable=False,
                ##kernel_initializer=tf.keras.initializers.constant([10]),
                ##bias_initializer=tf.keras.initializers.constant([-5])),
            ##])

    ### author's architecture -- overparameterized
    ##model = tf.keras.models.Sequential([
            ##tf.keras.layers.Conv2D(20, kernel_size=(3,3), padding='same', activation='relu'),
            ##tf.keras.layers.Conv2D(10, kernel_size=(1,1), padding='same', activation='relu'),
            ##tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation='sigmoid')
            ##])

    # author's architecture
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(2, kernel_size=(3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same', activation='sigmoid')
            ])

    # compile, train, evaluate model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(x_tr, y_tr, batch_size=100, epochs=20)
    model.evaluate(x_ts, y_ts)

    # show results
    viz.print_model_weights(model)
    viz.predict_glider(life, model)

