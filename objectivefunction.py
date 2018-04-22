import numpy as np

class FuncJ(object):
    def __init__(self, alpha, A, regularization_function):
        self._alpha = alpha
        self._A = A
        self._regularization_function = regularization_function

    def __call__(self, f, fmedido):
        return (np.sum((self._A(f) - fmedido)**2.0) +
                self._alpha*self._regularization_function(f))

    @property
    def alpha(self, alpha):
        ''' set alpha parameter '''
        self._alpha = alpha
