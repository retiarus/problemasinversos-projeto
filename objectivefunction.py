import numpy as np
from itertools import islice, count

class FuncJ(object):
    def __init__(self, alpha, A, regularizationFunction = None):
        self.__alpha = alpha
        self.__A = A
                 
        if regularizationFunction is None:
            self.__regularizationFunction = tikhonovorder0(A.shape[0])
        else:
            self.__regularizationFunction = regularizationFunction
            
    def __call__(self, f, fmedido):
        if self.J:
            return np.sum((self.__A(f) - fmedido)**2.0) + self.__alpha*self.__regularizationFunction(f)
        else:
            return None
        
    def changeAlpha(self, alpha):
        self.__alpha = alpha
