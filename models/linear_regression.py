import random
from typing import List

import sys
sys.path.append('..')
from autograd import Variable, Matrix
from optimizers import SGD
from dataloader import DataLoader
from loss_functions import mean_squared_error


class LinearRegression:
    """
    Implements linear regression.
    That is, we assume that the manifold of the data is a plane, which we can model
        by a linear combination of the values of each variable.
    """
    def __init__(self, params=None, bias=None, alpha=0, beta=0):
        """
        :param params: The starting parameters to use.
        :param bias: The starting bias to use.
        :param alpha: The amount to scale the sum of L1 norms in the regularizing term.
        :param beta: The amount to scale the sum of L2 norms in the regularizing term.
        """
        self.params = params
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def _evaluate(self, row: Matrix) -> Matrix:
        """
        Using self.params, evaluates row, which we assume to be an iterable
            filled with Variable objects.
        """
        return row * self.params + self.bias
        

    def _regularize(self) -> Matrix:
        """
        Returns the sum of the l1 and l2 norms of the parameters, scaled by alpha and beta.
        """
        return Matrix([[self.alpha]]) * self.params.sum() + self.beta * (self.params**2).sum()

    def fit(self, loader: DataLoader, optimizer=None, loss_function=None) -> None:
        """
        I should really be using a DataLoader here, to handle minibatching elegantly, 
            rather than using X, y like sklearn.

        Fits the model to the data.
        If no optimizer is passed in, the default optimizer is SGD.
        If no loss function is passed in, the default loss function is MSE.
        :returns: None; self.params are fit to the data.
        """
        if optimizer is None:
            optimizer = SGD(0.01)

        if loss_function is None:
            loss_function = mean_squared_error

        for X, y in loader:
            if self.params is None:
                self.params = Matrix([[Variable(random.random())] for _ in range(len(X[0]))])
                self.bias = Matrix([[Variable(random.random())]])

            output = self._evaluate(X)
            loss = loss_function(output, y)
            loss += self._regularize()
            self.params = optimizer.step(self.params, loss.get_grad(self.params))
            self.bias = optimizer.step(self.bias, loss.get_grad(self.bias))

    def predict(self, X: Matrix) -> Matrix:
        """
        Given X, returns a list of predictions for X.
        """
        preds = []
        for row in X:
            preds.append(self._evaluate(Matrix([row]))[0])
        return Matrix(preds)


if __name__ == "__main__":
    true_params = [0.3, 3, 5.4]
    f = lambda row: sum([row[i] * true_params[i] for i in range(len(row))]) + 3
    X = []
    y = []
    for _ in range(5000): 
        X.append([random.random() for _ in range(3)])
        y.append([f(X[-1])])

    loader = DataLoader(X, y)

    linreg = LinearRegression()
    optim = SGD(0.01, minibatch_size=5)
    linreg.fit(loader)
    print(linreg.params)
    print(linreg.bias)

    X_test = []
    for _ in range(10):
        X_test.append([random.random() for _ in range(3)])

    outputs = linreg.predict(Matrix(X_test))
    print(outputs)
