import math

from .linear_regression import LinearRegression

import sys
sys.path.append('.')
from autograd import Variable, Matrix
from optimizers import SGD
from dataloader import DataLoader
from loss_functions import binary_cross_entropy


def sigmoid(x: Variable) -> Variable:
    return 1 / (1 + math.e**(-x))

class LogisticRegression(LinearRegression):
    """
    Implements logistic regression.
    That is, we assume that the manifold of the data is a plane with a pointwise application of
        the sigmoid function.
    """
    def __init__(self, params=None, bias=None, alpha=0, beta=0):
        """
        :param params: The starting parameters to use.
        """
        super().__init__(params, bias, alpha, beta)

    def _evaluate(self, row: Matrix) -> Matrix:
        """
        Uses the _evaluate function in LinearRegression, then applies a sigmoid function
            elementwise to the resulting matrix.
        """
        mat = super()._evaluate(row).elementwise_apply(sigmoid)
        return mat

    def fit(self, loader: DataLoader, optimizer=None, loss_function=None) -> None:
        """
        Wrapper around LinearRegression's fit, but instead, we use binary cross entropy as 
            our default loss function.
        """
        loss_function = binary_cross_entropy if loss_function is None else loss_function
        super().fit(loader, optimizer, loss_function)
