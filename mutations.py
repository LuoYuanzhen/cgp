from collections import deque
from random import choice, random

from individual import Individual

class Move():
    """
    class holding a change
    contains two lists:
        - list of affected indicies
        - list of values to apply
    """
    def __init__(self, indices, changes):
        self.indicies = indices
        self.changes = changes

    def __eq__(self, other):
        if self.indicies != other.indicies:
            return False
        if self.changes != other.changes:
            return False
        return True



def point_mutation(individual):
    """
    perform a point mutation on given individual
    returns Move object
    """
    # handle case, when there is only one possible value
    # of gene at certain position
    indices = [i for i, (l, u) in enumerate(zip(*individual.bounds)) if u - l > 0]
    index = choice(indices)

    l_bounds = individual.bounds[0]
    u_bounds = individual.bounds[1]

    # construct the list of acceptable values
    possible_values = [x for x in range(
        l_bounds[index], u_bounds[index] + 1) if x != individual.genes[index]]

    return Move([index], [choice(possible_values)])


def single_mutation(individual):
    """ perform a 'single' mutation - mutate until active gene is changed """

    active_changed = False
    changed_indices = []
    changed_genes = []
    indices = [i for i, (l, u) in enumerate(zip(*individual.bounds)) if u - l > 0]
    l_bounds = individual.bounds[0]
    u_bounds = individual.bounds[1]

    while not active_changed:

        index = choice(indices)

        changed_indices.append(index)

        possible_values = [x for x in range(
                                        l_bounds[index],
                                        u_bounds[index] + 1)
                           if x != individual.genes[index]]

        changed_genes.append(choice(possible_values))

        # changing a gene from set of active genes

        if individual.active_gene(index):
            active_changed = True

    return Move(changed_indices, changed_genes)


def active_mutation(individual):
    """ Perform an active mutation - to-be mutated gene is chosen only
    from active genes """

    genes = individual.genes[:]
    l_bounds = individual.bounds[0]
    u_bounds = individual.bounds[1]
    agenes = individual.get_active_genes()

    # choose only genes with more than one possible value
    indices = []
    for gene_index in agenes:
        if u_bounds[gene_index] - l_bounds[gene_index] > 0:
            indices.append(gene_index)

    index = choice(indices)

    possible_values = [x for x in range(l_bounds[index], u_bounds[index] + 1) if x != genes[index]]

    return Move([index], [choice(possible_values)])


def probabilistic_mutation(individual, probability=0.25):
    """ Perform a probabilistic mutation - at each gene position there is a
    chance it will mutate """
    changed_indices = []
    new_values = []
    l_bounds = individual.bounds[0]
    u_bounds = individual.bounds[1]

    for index in range(0, len(individual.genes)):
        chance = random()
        if chance < probability:
            possible_values = [x for x in range(l_bounds[index], u_bounds[index] + 1)
                               if x != individual.genes[index]]
            if not possible_values:
                continue
            changed_indices.append(index)
            new_values.append(choice(possible_values))

    return Move(changed_indices, new_values)

MUTATIONS = {
    'point': point_mutation,
    'single': single_mutation,
    'active': active_mutation,
    'probabilistic': probabilistic_mutation
}
