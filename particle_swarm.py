
import copy
import numpy as np
import tqdm


class ParticleSwarm:
    '''Setup a Particle Swarm Optimzation'''

    def __init__(self, num_particles=100, inertia_weight=.5,
                 cognitive_weight=1, social_weight=2):
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def train(self, model, training_data, test_data=None, epochs=100, seed=0):
        '''Train a network on the provided data'''
        np.random.seed(seed)
        self.weights = model.get_weights()
        ##self.weights = [np.zeros_like(weight) for weight in model.get_weights()]
        self.setup_swarm(model, training_data)
        history = {'loss': [], 'val_loss': []}
        pbar = tqdm.trange(epochs, desc=f'Training')

        for _ in pbar:
            # update each particle
            self.update_swarm()
            self.evaluate_swarm(model, training_data)

            # evaluate the best model
            history['loss'].append(self.evaluate_best(model, training_data))
            pbar.set_postfix(loss=history['loss'][-1])
            if test_data is not None:
                history['val_loss'].append(self.evaluate_best(model, test_data))

        # return the best model
        model.set_weights(self.weights_best)

        return history

    def setup_swarm(self, model, training_data):
        '''Generate initial particles for searm optimization'''
        self.weights_best = copy.deepcopy(self.weights)
        self.error_best = self.evaluate_best(model, training_data)
        self.swarm = [{
            ##'weights': [20*np.random.rand(*weight.shape)-10 for weight in self.weights],
            ##'velocity': [2*np.random.rand(*weight.shape)-1 for weight in self.weights],
            'weights': [10*np.random.randn(*weight.shape) for weight in self.weights],
            'velocity': [np.random.randn(*weight.shape) for weight in self.weights],
            'error': float('inf'),
            'best_weights': copy.deepcopy(self.weights),
            'best_error': float('inf')}
            for _ in range(self.num_particles)]

    def update_swarm(self):
        '''Compute new position for each particle'''
        for particle in self.swarm:
            for i in range(len(particle['weights'])):
                # compute cognitive and social velocity
                cognitive_velocity = self.cognitive_weight\
                    * np.random.rand(*particle['weights'][i].shape)\
                    * (particle['best_weights'][i] - particle['weights'][i])
                social_velocity = self.social_weight\
                    * np.random.rand(*particle['weights'][i].shape)\
                    * (self.weights_best[i] - particle['weights'][i])
                # update particle velocity and position
                particle['velocity'][i] = self.inertia_weight * particle['velocity'][i]\
                    + cognitive_velocity + social_velocity
                particle['weights'][i] += particle['velocity'][i]
                particle['weights'][i] = np.clip(particle['weights'][i], -10, 10)

    def evaluate_swarm(self, model, training_data):
        '''Evaluate weight of each particle'''
        for particle in self.swarm:
            model.set_weights(particle['weights'])
            x, y = training_data
            z = model(np.array(x, ndmin=2).reshape(-1,*x.shape[1:])).numpy()
            particle['error'] = np.mean((y - z)**2)
            # determine if current particle weights is better
            if particle['error'] < particle['best_error']:
                particle['best_position'] = copy.deepcopy(particle['weights'])
                particle['best_error'] = particle['error']
                if particle['error'] < self.error_best:
                    self.weights_best = copy.deepcopy(particle['weights'])
                    self.error_best = particle['error']

    def evaluate_best(self, model, data):
        '''compute mse of the best model on the given data'''
        model.set_weights(self.weights_best)
        x, y = data
        z = model(np.array(x, ndmin=2).reshape(-1,*x.shape[1:])).numpy()
        mse = np.mean((y - z)**2)
        return mse

