import random

from ..autograd import Variable, Matrix
from ..linear_models import LinearRegression
from ..dataloader import DataLoader
from ..optimizers import SGD

def test_linreg():
    true_params = [0.3, 3, 5.4]
    f = lambda row: sum([row[i] * true_params[i] for i in range(len(row))]) + 3
    X = []
    y = []
    for _ in range(10000): 
        X.append([random.random() for _ in range(3)])
        y.append([f(X[-1])])

    loader = DataLoader(X, y)

    linreg = LinearRegression()
    optim = SGD(0.05, minibatch_size=5)
    linreg.fit(loader)

    eps = 1e-4

    for i in range(3):
        assert (linreg.params[i][0] - true_params[i]).abs() < eps

    assert (linreg.bias[0][0] - 3).abs() < eps

    X_test = []
    y_test = []
    for _ in range(10):
        X_test.append([random.random() for _ in range(3)])
        y_test.append(f(X_test[-1]))

    outputs = linreg.predict(Matrix(X_test))
    for i, output in enumerate(outputs):
        assert (output[0] - y_test[i]).abs() < eps
