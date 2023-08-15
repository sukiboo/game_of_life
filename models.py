import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def create_model_sequential(n, activation='relu', random_seed=0, name=None):
    """Construct sequential feedforward convolutional network to learn n-step GoF"""
    tf.keras.utils.set_random_seed(random_seed)
    layers = [tf.keras.Input((None,None,1))]
    for _ in range(n):
        layers.append(tf.keras.layers.Conv2D(
            2, kernel_size=(3,3), padding='same', activation=activation))
        layers.append(tf.keras.layers.Conv2D(
            1, kernel_size=(1,1), padding='same', activation='relu'))
    model = tf.keras.models.Sequential(layers)
    model.compile(loss='mse', metrics=['accuracy'])
    if name is not None:
        model._name = name
    return model


def create_model_recursive(n, activation='relu', random_seed=0, name=None):
    """Construct recursive feedforward convolutional network to learn n-step GoF"""
    tf.keras.utils.set_random_seed(random_seed)
    inputs = tf.keras.Input(shape=(None,None,1), name='input')
    conv1 = tf.keras.layers.Conv2D(
        2, kernel_size=(3,3), padding='same', activation=activation, name='conv1')
    conv2 = tf.keras.layers.Conv2D(
        1, kernel_size=(1,1), padding='same', activation='relu', name='conv2')
    outputs = inputs
    for _ in range(n):
        outputs = conv2(conv1(outputs))
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='mse', metrics=['accuracy'])
    if name is not None:
        model._name = name
    return model


def get_ground_truth_model(n, activation='relu'):
    """Create model that simulates n-step GoF"""
    model = create_model_sequential(n, activation=activation, name=f'{n}-step-GoF')
    if activation == 'relu':
        weights = n * [
            np.array([1,-1,1,-1,1,-1,1,-1,0,-1,1,-1,1,-1,1,-1,1,-1]).reshape(3,3,1,2),
            np.array([-3,3]), np.array([-1,-1]).reshape(1,1,2,1), np.array([1])]
    elif activation == 'tanh':
        weights = n * [
            np.array([1,1,1,1,1,1,1,1,.5,0,1,1,1,1,1,1,1,1]).reshape(3,3,1,2),
            np.array([-2.5,-3.5]), np.array([2,-2]).reshape(1,1,2,1), np.array([-1])]
    model.set_weights(weights)
    model.compile(loss='mse', metrics=['accuracy'])
    for layer in model.layers:
        layer.trainable = False
    return model


def train_model(model, alg, params_opt, params_train, data, random_seed=0):
    """Train model via backpropagation"""
    tf.keras.utils.set_random_seed(random_seed)
    optimizer = getattr(tf.keras.optimizers, alg.rstrip('0123456789'))(**params_opt)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)
    history = model.fit(*data, **params_train, verbose=0, callbacks=[tqdm_callback])
    return model, history.history

