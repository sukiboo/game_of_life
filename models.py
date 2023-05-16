
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from genetic_algorithm import GeneticAlgorithm
from evolution_strategy import EvolutionStrategy
from smoothing_optimization import SmoothingOptimization
from particle_swarm import ParticleSwarm



##def create_model_relu(n, random_seed=0, name=None):
    ##"""Construct feedforward convolutional network to learn n-step GoF"""
    ##tf.random.set_seed(random_seed)
    ##layers = [tf.keras.Input((None,None,1))]
    ##for _ in range(n):
        ##layers.append(tf.keras.layers.Conv2D(
            ##2, kernel_size=(3,3), padding='same', activation='relu'))
        ##layers.append(tf.keras.layers.Conv2D(
            ##1, kernel_size=(1,1), padding='same', activation='relu'))
    ##model = tf.keras.models.Sequential(layers)
    ##if name is not None:
        ##model._name = name
    ##return model

##def create_model(n, random_seed=0, name=None):
    ##"""Construct feedforward convolutional network to learn n-step GoF"""
    ##tf.random.set_seed(random_seed)
    ##layers = [tf.keras.Input((None,None,1))]
    ##for _ in range(n):
        ##layers.append(tf.keras.layers.Conv2D(
            ##2, kernel_size=(3,3), padding='same', activation='tanh'))
        ##layers.append(tf.keras.layers.Conv2D(
            ##1, kernel_size=(1,1), padding='same', activation='relu'))
    ##model = tf.keras.models.Sequential(layers)
    ##if name is not None:
        ##model._name = name
    ##return model


##def get_ground_truth_model(n):
    ##"""Create model that simulates n-step GoF"""
    ##model = create_model_relu(n, name=f'{n}-step-GoF')
    ##weights = n * [np.array([1,-1,1,-1,1,-1,1,-1,0,-1,1,-1,1,-1,1,-1,1,-1]).reshape(3,3,1,2),
                   ##np.array([-3,3]), np.array([-1,-1]).reshape(1,1,2,1), np.array([1])]
    ##model.set_weights(weights)
    ##for layer in model.layers:
        ##layer.trainable = False
    ##return model


def create_model_relu(n, random_seed=0, name=None):
    """same thing but repeating (conv1,conv2) n times"""
    tf.random.set_seed(random_seed)
    inputs = tf.keras.Input(shape=(None,None,1), name='input')
    conv1 = tf.keras.layers.Conv2D(
        2, kernel_size=(3,3), padding='same', activation='relu', name='conv1')
    conv2 = tf.keras.layers.Conv2D(
        1, kernel_size=(1,1), padding='same', activation='relu', name='conv2')
    outputs = inputs
    for _ in range(n):
        outputs = conv2(conv1(outputs))
    model = tf.keras.Model(inputs, outputs)
    if name is not None:
        model._name = name
    return model


def create_model(n, random_seed=0, name=None):
    """same thing but repeating (conv1,conv2) n times"""
    tf.random.set_seed(random_seed)
    inputs = tf.keras.Input(shape=(None,None,1), name='input')
    conv1 = tf.keras.layers.Conv2D(
        2, kernel_size=(3,3), padding='same', activation='tanh', name='conv1')
    conv2 = tf.keras.layers.Conv2D(
        1, kernel_size=(1,1), padding='same', activation='relu', name='conv2')
    outputs = inputs
    for _ in range(n):
        outputs = conv2(conv1(outputs))
    model = tf.keras.Model(inputs, outputs)
    if name is not None:
        model._name = name
    return model


def get_ground_truth_model(n):
    """Create model that simulates n-step GoF"""
    model = create_model_relu(n, name=f'{n}-step-GoF')
    weights = [np.array([1,-1,1,-1,1,-1,1,-1,0,-1,1,-1,1,-1,1,-1,1,-1]).reshape(3,3,1,2),
               np.array([-3,3]), np.array([-1,-1]).reshape(1,1,2,1), np.array([1])]
    model.set_weights(weights)
    model.compile(loss='mse', metrics=['accuracy'])
    for layer in model.layers:
        layer.trainable = False
    return model


def train_model_bp(n, data, epochs, params, random_seed=0):
    """Train model via backpropagation"""
    print('\n=== Backpropagation ===')
    model = create_model(n, name='model_bp', random_seed=random_seed)
    optimizer = getattr(tf.keras.optimizers, params['name'])(**params)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    (x_tr,y_tr), (x_ts,y_ts) = data
    model.evaluate(x_ts, y_ts)
    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    history = model.fit(x_tr, y_tr, epochs=epochs, verbose=0, callbacks=[tqdm_callback])
    model.evaluate(x_ts, y_ts)
    return model, history.history


def train_model_ps(n, data, epochs, params, random_seed=0):
    """Train model via particle swarm optimization"""
    print('\n=== Particle Swarm Optimization ===')
    model = create_model(n, name='model_ps', random_seed=random_seed)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    (x_tr,y_tr), (x_ts,y_ts) = data
    model.evaluate(x_ts, y_ts)
    ps = ParticleSwarm(**params)
    history = ps.train(model, (x_tr,y_tr), epochs=epochs)
    model.evaluate(x_ts, y_ts)
    return model, history


def train_model_ga(n, data, epochs, params, random_seed=0):
    """Train model via genetic algorithm"""
    print('\n=== Genetic Algorithm ===')
    model = create_model(n, name='model_ga', random_seed=random_seed)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    (x_tr,y_tr), (x_ts,y_ts) = data
    model.evaluate(x_ts, y_ts)
    ga = GeneticAlgorithm(**params)
    history = ga.train(model, (x_tr,y_tr), epochs=epochs)
    model.evaluate(x_ts, y_ts)
    return model, history


def train_model_es(n, data, epochs, params, random_seed=0):
    """Train model via evolution strategy"""
    print('\n=== Evolution Strategy ===')
    model = create_model(n, name='model_es', random_seed=random_seed)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    (x_tr,y_tr), (x_ts,y_ts) = data
    model.evaluate(x_ts, y_ts)
    es = EvolutionStrategy(**params)
    history = es.train(model, (x_tr,y_tr), epochs=epochs)
    model.evaluate(x_ts, y_ts)
    return model, history


def train_model_so(n, data, epochs, params, random_seed=0):
    """Train model via smoothing-based optimization"""
    print('\n=== Smoothing-Based Optimization ===')
    model = create_model(n, name='model_so', random_seed=random_seed)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    (x_tr,y_tr), (x_ts,y_ts) = data
    model.evaluate(x_ts, y_ts)
    so = SmoothingOptimization(**params)
    history = so.train(model, (x_tr,y_tr), epochs=epochs)
    model.evaluate(x_ts, y_ts)
    return model, history

