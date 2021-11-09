import numpy as np


def protected_division(x0, x1):
    return np.divide(x0, x1, out=np.copy(x0), where=x1!=0)


def protected_sqrt(x0):
    return np.sqrt(x0, out=np.zeros_like(x0), where=x0>=0)


class BaseFunction:
    def __init__(self, func, arity, symbol):
        self.func = func
        self.arity = arity
        self.symbol = symbol

    def __call__(self, *args):
        return self.func(*args)

    def expr(self, *operants):
        if self.arity == 1:
            return "{}({})".format(self.symbol, operants[0])
        if self.arity == 2:
            return "({}{}{})".format(operants[0], self.symbol, operants[1])


class PowerFunction(BaseFunction):
    def __init__(self, func, symbol, arity=1):
        super(PowerFunction, self).__init__(func, arity, symbol)

    def expr(self, *operants):
        return "({})^{}".format(operants[0], self.symbol)


func_map = {
    'add': (BaseFunction(np.add, 2, '+'), 2),
    'sub': (BaseFunction(np.subtract, 2, '-'), 2),
    'mul': (BaseFunction(np.multiply, 2, '*'), 2),
    'div': (BaseFunction(protected_division, 2, '/'), 2),
    'sin': (BaseFunction(np.sin, 1, 'sin'), 1),
    'cos': (BaseFunction(np.cos, 1, 'cos'), 1),
    'tan': (BaseFunction(np.tan, 1, 'tan'), 1),
    'sqrt': (PowerFunction(protected_sqrt, '(1/2)'), 1),
    'square': (PowerFunction(np.square, '2'), 1)
}


default_function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt', 'square']


