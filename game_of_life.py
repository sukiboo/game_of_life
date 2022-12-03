
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
    """

    def __init__(self, show_model=True):
        """Initialize class variables"""
        self.setup_model()
        if show_model:
            self.model.summary()
            self.print_model_weights()
        self.setup_predefined_states()
        os.makedirs('./data/', exist_ok=True)

    def setup_model(self):
        """Create the Game-of-Life model"""
        self.model = tf.keras.models.Sequential(name='game_of_life')
        self.model.add(tf.keras.Input(shape=(None,None,1)))
        const = tf.keras.initializers.constant
        # convolutional layer
        self.model.add(tf.keras.layers.Conv2D(
            filters=2, kernel_size=(3,3), padding='same',
            name='conv', trainable=False, dynamic=True,
            kernel_initializer=const([0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,1]),
            bias_initializer=const([0,0])))
        # fully connected layer 1
        self.model.add(tf.keras.layers.Dense(
            units=2, activation='relu',
            name='fc1', trainable=False, dynamic=True,
            kernel_initializer=const([0,-1,1,-1]),
            bias_initializer=const([-3,3])))
        # fully connected layer 2
        self.model.add(tf.keras.layers.Dense(
            units=1, activation='relu',
            name='fc2', trainable=False, dynamic=True,
            kernel_initializer=const([-1,-1]),
            bias_initializer=const([1])))
        # initialize model weights
        self.state = [[0]]
        self.step()

    def print_model_weights(self):
        """Display the network architecture and weights"""
        layers = self.model.layers
        print('\nconv weights:')
        print(np.squeeze(layers[0].get_weights()[0]).transpose(2,0,1))
        print('\nfc1 weights:')
        print(layers[1].get_weights()[0], layers[1].get_weights()[1], sep='\n ')
        print('\nfc2: weights')
        print(layers[2].get_weights()[0], layers[2].get_weights()[1], sep='\n ')

    def play(self, init=0, board_size=(32,32), steps=100, name=None, animate=True,
             cmap=clr.ListedColormap(['#ffffff','#f06923'])):
        """Play a simulation of Game of Life"""
        self.name = name if isinstance(name, str)\
            else time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime())
        os.makedirs('./data/' + self.name, exist_ok=True)
        state_path = './data/{:s}/{:0' + str(len(str(steps))) + 'd}.png'
        # run game
        self.setup_initial_state(init=init, board_size=board_size)
        for i in range(steps):
            img.imsave(state_path.format(self.name, i), self.state, cmap=cmap)
            if not self.terminal:
                self.step()
            else:
                break
        # generate gif
        if animate:
            self.animate_game()

    def step(self):
        """Advance time in a simulation by one step"""
        prev_state = self.state
        self.state = np.squeeze(self.model(np.expand_dims(self.state, axis=(0,-1))).numpy())
        self.terminal = (np.sum(np.abs(self.state - prev_state)) == 0)

    def setup_initial_state(self, init, board_size):
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
        elif isinstance(init, np.ndarray) and init.ndim == 2:
            self.state = (init > (np.min(init) + np.max(init)) / 2).astype(int)
        # provided image path
        else:
            img = 1 - io.imread(init, as_gray=True)
            self.state = (img > (np.min(img) + np.max(img)) / 2).astype(int)

    def setup_predefined_states(self):
        """Create a dictionary of predefined initial states"""
        self.predefined_states = {
            'glider': [(9,9), (0,0), (1,1), (1,2), (2,0), (2,1)],
            'lwss': [(9,27), (1,1), (1,4), (2,5), (3,1), (3,5), (4,2), (4,3), (4,4), (4,5)],
            'mwss': [(9,27), (1,3), (2,1), (2,5), (3,6), (4,1), (4,6), (5,2), (5,3), (5,4),
            (5,5), (5,6)],
            'hwss': [(9,27), (1,3), (1,4), (2,1), (2,6), (3,7), (4,1), (4,7), (5,2), (5,3),
            (5,4), (5,5), (5,6), (5,7)],
            'pulsar': [(15,15), (1,3), (1,4), (1,5), (1,9), (1,10), (1,11), (3,1), (3,6),
            (3,8), (3,13), (4,1), (4,6), (4,8), (4,13), (5,1), (5,6), (5,8), (5,13), (6,3),
            (6,4), (6,5), (6,9), (6,10), (6,11), (8,3), (8,4), (8,5), (8,9), (8,10), (8,11),
            (9,1), (9,6), (9,8), (9,13), (10,1), (10,6), (10,8), (10,13), (11,1), (11,6),
            (11,8), (11,13), (13,3), (13,4), (13,5), (13,9), (13,10), (13,11)],
            'pentadecathlon': [(11,18), (4,6), (4,11), (5,4), (5,5), (5,7), (5,8), (5,9),
            (5,10), (5,12), (5,13), (6,6), (6,11)]}

    def animate_game(self):
        """Create an animated gif of the simulation"""
        states = ['./data/' + self.name + '/000.png']*23\
                 + sorted(glob.glob('./data/' + self.name + '/*.png'))
        writer = imageio.get_writer('./data/{:s}.gif'.format(self.name), mode='I', duration=1/12)
        with writer as gif:
            for state in states:
                gif.append_data(imageio.imread(state))

    def generate_dataset(self, board_size=(32,32), num_sim=100, steps=100, animate=False):
        """Create a dataset consisting of multiple simulations"""
        for sim in range(num_sim):
            name = 'game' + '_{:d}x{:d}_'.format(*board_size)\
                   + ('{:0' + str(len(str(steps))) + 'd}').format(sim)
            self.play(init=sim, board_size=board_size, steps=steps, name=name, animate=animate)


if __name__ == '__main__':
    # run a sample simulation
    life = GameOfLife(show_model=True)
    ##life.play(init='glider', name='glider')
    ##life.play(init='./lirio.png', name='lirio', steps=400)

    ##life.play(init=0, board_size=(9,9), steps=100, animate=True)
    ##life.play(init='pulsar', steps=100, name='pulsar', animate=True)
    ##life.play(init='./input.png', steps=100, animate=True)

    ##life.generate_dataset(num_sim=10, steps=100, animate=True)





