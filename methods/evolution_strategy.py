import copy
import numpy as np
import tqdm


class EvolutionStrategy:
    '''Setup an evolution strategy'''

    def __init__(self, name='es', learning_rate=1e-3, population=100, sigma=.1):
        self.learning_rate = learning_rate
        self.population = population
        self.sigma = sigma

    def train(self, model, training_data, test_data=None, epochs=100, seed=0):
        '''Train a network on the provided data'''
        np.random.seed(seed)
        self.weights = model.get_weights()
        history = {'loss': [], 'val_loss': []}
        pbar = tqdm.trange(epochs, desc=f'Training')

        for _ in pbar:
            # mutate, evaluate, truncate
            self.mutate_parents()
            self.compute_fitness(model, training_data)
            self.truncate_population()

            # evaluate the best model
            history['loss'].append(self.evaluate_best(model, training_data))
            pbar.set_postfix(loss=history['loss'][-1])
            if test_data is not None:
                history['val_loss'].append(self.evaluate_best(model, test_data))

        # return the best model
        model.set_weights(self.weights)

        return history

    def mutate_parents(self):
        '''Create a list of perturbed network weights'''
        self.model_weights = []
        for i in range(self.population):
            weights = copy.deepcopy(self.weights)
            for weight in weights:
                weight += self.sigma * np.random.randn(*weight.shape)
            self.model_weights.append(weights)

    def compute_fitness(self, model, training_data):
        '''Compute fitness score for each weight -- I use 1-2*mse'''
        x, y = training_data
        self.model_fitness = []
        for weights in self.model_weights:
            model.set_weights(weights)
            z = model(np.array(x, ndmin=2).reshape(-1,*x.shape[1:])).numpy()
            fitness = 1 - 2*np.mean((y - z)**2)
            self.model_fitness.append(fitness)

    def truncate_population(self):
        '''Update weights based on their fitness'''
        for weights, fitness, eps in zip(self.model_weights, self.model_fitness, self.eps):
            for i in range(len(self.weights)):
                change = (weights[i] - self.weights[i]) / self.population
                self.weights[i] += self.learning_rate * change * fitness / self.sigma

    def evaluate_best(self, model, data):
        '''Compute mse of the best model on the given data'''
        model.set_weights(self.weights)
        x, y = data
        z = model(np.array(x, ndmin=2).reshape(-1,*x.shape[1:])).numpy()
        mse = np.mean((y - z)**2)
        return mse

