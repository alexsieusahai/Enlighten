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
        mat = super()._evaluate(row)
        return mat.elementwise_apply(sigmoid)


if __name__ == "__main__":
    import random
    true_params = [0.3, 0.7, 0.2]
    f = lambda row: sigmoid(sum([row[i] * true_params[i] for i in range(len(row))]) + 0.3)
    X = []
    y = []
    for _ in range(2000):
        X.append([random.random() for _ in range(3)])
        y.append([f(X[-1])])

    loader = DataLoader(X, y)

    linreg = LogisticRegression()
    optim = SGD(0.01, minibatch_size=5)
    linreg.fit(loader)
    print(linreg.params)
    print(linreg.bias)

    X_test = []
    for _ in range(10):
        X_test.append([random.random() for _ in range(3)])

    outputs = linreg.predict(Matrix(X_test))
    print(outputs)
