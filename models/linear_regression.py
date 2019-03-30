import random
from typing import List

import sys
sys.path.append('..')
from autograd import Variable, Matrix
from optimizers import SGD
from dataloader import DataLoader


class LinearRegression:
    """
    Implements linear regression.
    That is, we assume that the manifold of the data is a plane, which we can model
        by a linear combination of the values of each variable.
    """
    def __init__(self, params=None):
        """
        :param params: The starting parameters to use.
        """
        self.params = params


    def _evaluate(self, row: List[Variable]) -> Variable:
        """
        Using self.params, evaluates row, which we assume to be an iterable
            filled with Variable objects.
        """
        return row * self.params


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
            # should import it from some loss functions directory or something in the future
            loss_function = lambda x, y: (x-y).abs()

        if self.params is None:
            self.params = Matrix([[Variable(random.random())] for _ in range(len(X[0]))])

        # without minibatching, I treat this is matrix mult; I should switch to matrix mult,
        #     more general
        for X_batch, y_batch in loader:
            output = self._evaluate(X_batch)
            loss = loss_function(output, y_batch)
            self.params = optimizer.step(self.params, loss.get_grad(self.params))


    def predict(self, test_loader: DataLoader) -> List[Variable]:
        """
        Given X, returns a list of predictions for X.
        """
        return [self._evaluate(X) for X in test_loader]

if __name__ == "__main__":
    true_params = [0.3, 10, 5.4]
    f = lambda row: sum([row[i] * true_params[i] for i in range(len(row))])
    X = []
    y = []
    for _ in range(1000):
        X.append([random.random() for _ in range(3)])
        y.append(f(X[-1]))

    loader = DataLoader(X, y, minibatch_size=1)

    linreg = LinearRegression()
    linreg.fit(loader)
    print(linreg.params)

    X_test = []
    for _ in range(10):
        X_test.append([random.random() for _ in range(3)])

    test_loader = DataLoader(X_test)
    outputs = linreg.predict(test_loader)
    for output in outputs:
        print(output[0][0].value)
