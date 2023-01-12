
import copy
import numpy as np
import tqdm


class GeneticAlgorithm:
    '''Setup a genetic algorithm'''

    def __init__(self, population=100, parents=10, sigma=.1):
        self.population = population
        self.parents = parents
        self.sigma = sigma

    def train(self, model, training_data, test_data=None, epochs=100, seed=0):
        '''Train a network on the provided data'''
        np.random.seed(seed)
        self.model_weights = [model.get_weights()]
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
        model.set_weights(self.model_weights[0])

        return history

    def mutate_parents(self):
        '''Create a list of perturbed network weights'''
        i, p = 0, len(self.model_weights)
        while len(self.model_weights) < self.population:
            weights = copy.deepcopy(self.model_weights[i % p])
            for weight in weights:
                weight += self.sigma * np.random.randn(*weight.shape)
            i += 1
            self.model_weights.append(weights)

    def compute_fitness(self, model, training_data):
        '''Compute fitness score for each weight -- I use 1 - relative mse'''
        x, y = training_data
        self.model_fitness = []
        for weights in self.model_weights:
            model.set_weights(weights)
            z = model(np.array(x, ndmin=2).reshape(-1,*x.shape[1:])).numpy()
            fitness = 1 - np.mean((y - z)**2) / np.mean(y**2)
            self.model_fitness.append(fitness)

    def truncate_population(self):
        '''Select and keep the fittest weights'''
        best_inds = np.argsort(self.model_fitness)[:self.parents:-1]
        self.model_weights = [self.model_weights[i] for i in best_inds]

    def evaluate_best(self, model, data):
        '''compute mse of the best model on the given data'''
        model.set_weights(self.model_weights[0])
        x, y = data
        z = model(np.array(x, ndmin=2).reshape(-1,*x.shape[1:])).numpy()
        mse = np.mean((y - z)**2)
        return mse

