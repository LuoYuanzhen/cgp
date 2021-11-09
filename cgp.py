import numpy as np

from sklearn.metrics import mean_squared_error

from functions import func_map, default_function_set
from parameters import FunctionSet, Parameters
from search import simple_es


class MyCGPModel:
    def __init__(self, param_dict):
        """

        :param param_dict (dict type): should contain n_inputs, n_outputs, n_rows, n_columns, function_set,

        """
        # get function set
        self.function_set = FunctionSet()
        for func in param_dict['function_set']:
            assert func in func_map
            self.function_set.add(*func_map[func])

        self.params = Parameters(n_inputs=param_dict['n_inputs'],
                                 n_outputs=param_dict['n_outputs'],
                                 n_rows=param_dict['n_rows'],
                                 n_columns=param_dict['n_columns'],
                                 function_set=self.function_set)
        self.pop = None

    def train(self, X, y, cost_function=mean_squared_error,
              population_size=10, n_generations=500,
              mutation='probabilistic', mutation_propability=0.25,
              verbose=False
              ):
        # population_size, n_generations, mutation_probability, verbose
        population = simple_es(X, y,
                               cost_function=cost_function,
                               params=self.params,
                               population_size=population_size,
                               evaluations=n_generations*population_size,
                               mutation=mutation,
                               mutation_probability=mutation_propability,
                               verbose=verbose)
        self.pop = population

    def __call__(self, X):
        if self.pop is None:
            raise UserWarning("Don't use __call__ before training.")
        elite = self.pop[0]
        return elite.transform(X)

    def get_expression(self):
        return self.pop[0].get_expression()


if __name__ == '__main__':

    from sklearn.datasets import fetch_california_housing

    dataset = fetch_california_housing()
    X_train = dataset.data
    y_train = dataset.target

    param_dict = {
        'n_inputs': X_train.shape[1],
        'n_outputs': 1,
        'n_rows': 2,
        'n_columns': 50,
        'function_set': default_function_set
    }

    cgp = MyCGPModel(param_dict)
    cgp.train(X_train, y_train, n_generations=500, verbose=1)

    print(mean_squared_error(cgp(X_train), y_train))
    print(cgp.get_expression())
