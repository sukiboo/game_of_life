import numpy as np
import tqdm


class SmoothingOptimization:
    '''Setup smoothing-based optimization'''

    def __init__(self, name='dgs', learning_rate=1e-3, sigma=.1, num_quad=5):
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.num_quad = num_quad

    def weights_to_array(self, Weights):
        '''transform a nested list of weights into a flat array'''
        array = np.concatenate([weights.flatten() for weights in Weights])
        return array

    def array_to_weights(self, array):
        '''transform a flat array into a nested list of weights'''
        arrays = np.split(array, np.cumsum(self.sizes)[:-1])
        Weights = [arrays[i].reshape(self.shapes[i]) for i in range(len(arrays))]
        return Weights

    def train(self, model, training_data, test_data=None, epochs=100):
        '''Train a network on the provided data'''
        self.weights = model.get_weights()
        self.shapes = [weights.shape for weights in self.weights]
        self.sizes = [weights.size for weights in self.weights]
        history = {'loss': [], 'val_loss': []}
        pbar = tqdm.trange(epochs, desc=f'Training')

        P, W = np.polynomial.hermite.hermgauss(self.num_quad)
        dim = sum(self.sizes)
        u = np.eye(dim)
        for _ in pbar:
            weights_array = self.weights_to_array(self.weights)

            # estimate smoothed gradient along each direction
            dg = np.zeros(dim)
            for d in range(dim):
                # define directional function
                g = lambda t: self.compute_loss(model, weights_array + t*u[d], training_data)
                # estimate smoothed gradient
                g_vals = np.array([g(self.sigma * p) for p in P])
                dg[d] = np.sum(W * P * g_vals) / (self.sigma * np.sqrt(np.pi)/2)

            # reduce sigma
            self.sigma = max(.999*self.sigma, 1e-4)

            self.weights = self.array_to_weights(weights_array\
                                                 - self.learning_rate * np.matmul(dg, u))
            model.set_weights(self.weights)

            # evaluate the best model
            history['loss'].append(self.evaluate_model(model, training_data))
            pbar.set_postfix(loss=history['loss'][-1])
            if test_data is not None:
                history['val_loss'].append(self.evaluate_model(model, test_data))

        return history

    def compute_loss(self, model, weights_array, training_data):
        '''Compute mse loss of the model with the given array of weights'''
        weights = self.array_to_weights(weights_array)
        model.set_weights(weights)
        x, y = training_data
        z = model(np.array(x, ndmin=2).reshape(-1,*x.shape[1:])).numpy()
        mse = np.mean((y - z)**2)
        return mse

    def evaluate_model(self, model, data):
        '''Evaluate model on a given data'''
        x, y = data
        z = model(np.array(x, ndmin=2).reshape(-1,*x.shape[1:])).numpy()
        loss = np.mean((y - z)**2)
        return loss

