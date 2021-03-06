"""Search strategies."""

from array import array
from collections import deque
import random
from functools import partial

import numpy as np

from mutations import MUTATIONS
from individual import IndividualBuilder
from parameters import FunctionSet, Parameters
from utils import handle_invalid_decorator, UnknownMutationException

@handle_invalid_decorator
def simple_es(X, y, cost_function, params,
              target_fitness=None,
              population_size=5,
              evaluations=5000,
              random_state=None,
              mutation='probabilistic',
              mutation_probability=0.25,
              verbose=False,
              log=None,
              seed_individual=None):
    """Optimize a CGP system using a simple evolutionary strategy.

    Args:
        X (numpy.ndarray): input data, number of columns have to match the n_inputs
            parameter of Parameters object
        y (numpy.ndarray): target output data, number of columns have to match
            the n_outputs parameter of Parameters object
        cost_function (callable): cost function to minimize. It has two
            arguments `(y_true, y_pred)`, where `y_true` is target output data
            and `y_pred` is output of CGP individual
        params (Parameters): instance of Parameters class
        target_fitness (number or None): fitness, at which evolution will stop. If None, it is not considered
        population_size (int): size of population including parent
        evaluations (int): maximum number of cost function evaluations
        random_state (int): seed for random number generator
        mutation (string): type of mutation to use, accept values `'point',
            'active', 'single', 'probabilistic'`
        mutation_probability (float): probability of mutating a given gene (
            used only when `mutation` argument is set to `'probabilistic'`
        verbose (bool): if `True`, outputs evolution info every 100 generations
        log (list): if provided with a list, best fitness of each generation
            is stored here
        seed_individual (Individual): if provided with instance of Individual class,
            the initial population is created according to this object - parent of first
            generation.

    Returns:
        List of individuals, the last generation of evolution
    """

    if mutation not in MUTATIONS:
        raise UnknownMutationException("Provided type of mutation is not implemented.")

    move = MUTATIONS[mutation]
    if mutation == 'probabilistic':
        move = partial(move, probability=mutation_probability)

    if random_state:
        random.seed(random_state)

    # initial generation
    ib = IndividualBuilder(params)

    if seed_individual:
        population = [seed_individual.apply(move(seed_individual)) for _ in range(population_size - 1)]
        population += [seed_individual]
    else:
        population = [ib.create() for _ in range(population_size)]

    n_evals = 0

    generation = 0

    for individual in population:
        if params.cf_individual:
            individual.fitness = cost_function(y, individual)
        else:
            output = individual.transform(X)
            individual.fitness = cost_function(y, output)
        n_evals += 1


    while n_evals < evaluations:
        generation += 1

        parent = min(population, key=lambda x: x.fitness)

        if log is not None:
            log.append(parent.fitness)

        if target_fitness is not None and parent.fitness <= target_fitness:
            population.sort(key=lambda x: x.fitness)
            return population

        population = [parent.apply(move(parent)) for _ in range(population_size - 1)]

        population += [parent]

        for individual in population:
            if params.cf_individual:
                individual.fitness = cost_function(y, individual)
            else:
                output = individual.transform(X)
                individual.fitness = cost_function(y, output)
            n_evals += 1

        if verbose and generation % verbose == 0:
            print(f'Gen: {generation}, population: {sorted([x.fitness for x in population])}')
            print("\t Best individual:{}".format(population[0].get_expression()))


    population.sort(key=lambda x: x.fitness)

    if log is not None:
        log.append(population[0].fitness)

    return population

