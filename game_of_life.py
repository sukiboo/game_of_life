
import os
import glob
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.colors as clr
import imageio
from skimage import io


class GameOfLife:
    """
    Create a network that simulates Conway's Game of Life
    See full documentation at https://github.com/sukiboo/game_of_life
    UPD: there's no such repo, not sure why I wrote that
    UPDUPD: alright I created the repo
    """

    def __init__(self, show_model=False, colors=['#ffffff','#f06923']):
        """Initialize class variables"""
        self.setup_model()
        if show_model:
            self.model.summary()
            self.print_model_weights()
        self.setup_predefined_states()
        self.cmap = clr.ListedColormap(colors)

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
        self.state = np.eye(3)
        self.step()

    def print_model_weights(self):
        """Display the network architecture and weights"""
        layers = self.model.layers
        print('\nconv1 weights:')
        print(np.squeeze(layers[0].get_weights()[0]).transpose(2,0,1))
        print('\nconv1 biases:')
        print(' ', layers[0].get_weights()[1])
        print('\nconv2 weights:')
        print(np.squeeze(layers[1].get_weights()[0]).transpose(2,0,1))
        print('\nconv2 biases:')
        print(' ', layers[1].get_weights()[1])

    def step(self):
        """Advance time in a simulation by one step"""
        prev_state = self.state
        self.state = np.squeeze(self.model(np.expand_dims(self.state, axis=(0,-1))).numpy())
        self.terminal = (np.sum(np.abs(self.state - prev_state)) == 0)

    def setup_state(self, init=0, board_size=(32,32)):
        """Setup an initial state of a simulation"""
        self.terminal = False
        # predefined states
        if init in self.predefined_states:
            self.state = np.zeros(self.predefined_states[init][0])
            self.state[tuple(zip(*self.predefined_states[init][1:]))] = 1
        # random state
        elif isinstance(init, int):
            np.random.seed(init)
            self.state = np.random.randint(2, size=board_size)
        # provided numpy array
        elif isinstance(init, np.ndarray) and (init.ndim == 2):
            self.state = (init > (np.min(init) + np.max(init)) / 2).astype(int)
        # provided image path
        else:
            img = 1 - io.imread(init, as_gray=True)
            self.state = (img > (np.min(img) + np.max(img)) / 2).astype(int)

    def play(self, steps=100, name=None):
        """Play a simulation of Game of Life"""
        name = name if name is not None else int(time.time())
        os.makedirs(f'./data/{name}', exist_ok=True)
        # run game
        for step in range(steps):
            img.imsave(f'./data/{name}/{step:03d}.png', self.state, cmap=self.cmap)
            if not self.terminal:
                self.step()
            else:
                break

    def setup_predefined_states(self):
        """Create a dictionary of predefined initial states"""
        self.predefined_states = {
            'glider': [(8,8), (0,0), (1,1), (1,2), (2,0), (2,1)],
            'lwss': [(9,27), (1,1), (1,4), (2,5), (3,1), (3,5), (4,2), (4,3), (4,4), (4,5)],
            'mwss': [(9,27), (1,3), (2,1), (2,5), (3,6), (4,1), (4,6), (5,2), (5,3), (5,4),
                     (5,5), (5,6)],
            'hwss': [(9,27), (1,3), (1,4), (2,1), (2,6), (3,7), (4,1), (4,7), (5,2), (5,3),
                     (5,4), (5,5), (5,6), (5,7)],
            'pulsar': [(15,15), (1,3), (1,4), (1,5), (1,9), (1,10), (1,11), (3,1), (3,6),
                       (3,8), (3,13), (4,1), (4,6), (4,8), (4,13), (5,1), (5,6), (5,8),
                       (5,13), (6,3), (6,4), (6,5), (6,9), (6,10), (6,11), (8,3), (8,4),
                       (8,5), (8,9), (8,10), (8,11), (9,1), (9,6), (9,8), (9,13), (10,1),
                       (10,6), (10,8), (10,13), (11,1), (11,6), (11,8), (11,13), (13,3),
                       (13,4), (13,5), (13,9), (13,10), (13,11)],
            'pentadecathlon': [(11,18), (4,6), (4,11), (5,4), (5,5), (5,7), (5,8), (5,9),
                               (5,10), (5,12), (5,13), (6,6), (6,11)]}

    def animate_game(self, name):
        """Create an animated gif of the simulation"""
        states = 23*[f'./data/{name}/000.png'] + sorted(glob.glob(f'./data/{name}/*.png'))
        writer = imageio.get_writer(f'./data/{name}.gif', mode='I', duration=1/12)
        with writer as gif:
            for state in states:
                gif.append_data(imageio.imread(state))

    def generate_dataset_old(self, board_size=(32,32), num_sim=10, steps=100):
        """Create a dataset consisting of multiple simulations"""
        for sim in range(num_sim):
            name = f'datasets/{board_size[0]}x{board_size[1]}_{sim}'
            self.setup_state(init=sim, board_size=board_size)
            self.play(steps=steps, name=name)

    def generate_dataset(self, step=1, board_size=(32,32), num=100, random_seed=0):
        """Create a dataset consisting of multiple simulations"""
        np.random.seed(random_seed)
        X = np.random.randint(2, size=(num,*board_size,1)).astype(np.float32)
        Y = X.copy()
        for _ in range(step):
            Y = self.model(Y).numpy()
        return (X, Y)


if __name__ == '__main__':

    life = GameOfLife(show_model=True)
    x, y = life.generate_dataset()

    ##life.setup_state(init='glider')
    ##life.play(name='glider')
    ##life.animate_game(name='glider')

