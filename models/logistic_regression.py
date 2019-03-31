import math

from linear_regression import LinearRegression

import sys
sys.path.append('..')
from autograd import Variable, Matrix
from optimizers import SGD
from dataloader import DataLoader


def sigmoid(x: Variable) -> Variable:
    return 1 / (1 + math.e**(-x))

class LogisticRegression(LinearRegression):
    """
    Implements logistic regression.
    That is, we assume that the manifold of the data is a plane with a pointwise application of
        the sigmoid function.
    """
    def __init__(self, params=None):
        """
        :param params: The starting parameters to use.
        """
        self.params = params

    def _evaluate(self, row: Matrix) -> Matrix:
        """
        Uses the _evaluate function in LinearRegression, then applies a sigmoid function
            elementwise to the resulting matrix.
        """
        mat = super()._evaluate(row).elementwise_apply(sigmoid)
        return mat


if __name__ == "__main__":
    import random

    from loss_functions import binary_cross_entropy

    params = [0.3, 0.6, 0.2]
    f = lambda row: sigmoid(sum([row[i] * params[i] for i in range(len(row))]) + 0.2)
    X = []
    y = []
    thresh = 0.3
    for _ in range(2000):
        X.append([random.random() for _ in range(3)])
        val = f(X[-1])
        print(val)
        y.append([val > thresh])

    loader = DataLoader(X, y)

    logreg = LogisticRegression()
    optim = SGD(0.01, minibatch_size=5)
    logreg.fit(loader)
    print(logreg.params)
    print(logreg.bias)
    print(binary_cross_entropy(logreg.predict(Matrix(X)), Matrix(y)))

    X_test = []
    for _ in range(10):
        X_test.append([random.random() for _ in range(3)])

    outputs = logreg.predict(Matrix(X_test))
    for i, output in enumerate(outputs):
        print(output, f(X_test[i]) > thresh)
