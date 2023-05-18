import numpy as np
import tensorflow as tf


class GameOfLife:
    """Create a network that simulates Conway's Game of Life"""

    def __init__(self, show_model=False):
        """Initialize class variables"""
        self.setup_model()
        if show_model:
            self.model.summary()
            self.print_model_weights()

    def setup_model(self):
        """Create the Game-of-Life model"""
        const = tf.keras.initializers.constant
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(None,None,1)),
            # convolutional layer 1
            tf.keras.layers.Conv2D(filters=2, kernel_size=(3,3), padding='same',
                name='conv1', trainable=False, dynamic=True, activation='relu',
                kernel_initializer=const([1,-1,1,-1,1,-1,1,-1,0,-1,1,-1,1,-1,1,-1,1,-1]),
                bias_initializer=const([-3,3])),
            # convolutional layer 2
            tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), padding='same',
                name='conv2', trainable=False, dynamic=True, activation='relu',
                kernel_initializer=const([-1,-1]),
                bias_initializer=const([1]))],
            name='game_of_life')
        # initialize model weights
        self.predict(np.zeros((1,3,3,1)))

    def predict(self, X, steps=1):
        """Compute the board state after the given number of steps"""
        Y = X.copy()
        for _ in range(steps):
            Y = self.model(Y)
        return Y.numpy()

    def print_model_weights(self):
        """Display the network architecture and weights"""
        layers = self.model.layers
        print('\nconv1 weights:')
        print(np.squeeze(layers[0].get_weights()[0]).transpose(2,0,1))
        print('\nconv1 biases:')
        print(' ', layers[0].get_weights()[1])
        print('\nconv2 weights:')
        print(np.squeeze(layers[1].get_weights()[0]))
        print('\nconv2 biases:')
        print(' ', layers[1].get_weights()[1])

    def generate_dataset(self, num=100, steps=1, board_size=(32,32),
                         density=.5, random_seed=None):
        """Create a dataset consisting of multiple simulations"""
        if random_seed:
            np.random.seed(random_seed)
        X = np.zeros(num * np.prod(board_size))
        num_cells = X.size
        num_alive = int(num_cells * density)
        alive_cells = np.random.choice(num_cells, size=num_alive, replace=False)
        X[alive_cells] = 1
        X = X.reshape(num, *board_size, 1).astype(np.float32)
        Y = self.predict(X, steps=steps)
        return (X, Y)


if __name__ == '__main__':

    life = GameOfLife(show_model=True)
    x, y = life.generate_dataset(steps=2, num=1)

