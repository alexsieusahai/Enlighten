import numpy as np

import random

from ..autograd import Variable, Matrix
from ..linear_models import LogisticRegression, sigmoid
from ..dataloader import DataLoader
from ..optimizers import SGD
from ..loss_functions import binary_cross_entropy


def test_logreg():

    params = [0.3, 0.6, 0.2]
    f = lambda row: sigmoid(sum([row[i] * params[i] for i in range(len(row))]) + 0.2)
    X = []
    y = []
    thresh = 0.3
    for _ in range(2000):
        X.append([random.random() for _ in range(3)])
        val = f(X[-1])
        y.append([val > thresh])

    loader = DataLoader(X, y)

    logreg = LogisticRegression()
    optim = SGD(0.01, minibatch_size=5)
    logreg.fit(loader)
    assert binary_cross_entropy(logreg.predict(Matrix(X)), Matrix(y))[0][0] < 0.1

    X_test = []
    for _ in range(10):
        X_test.append([random.random() for _ in range(3)])

    outputs = logreg.predict(Matrix(X_test))
    res = []

    def proba_agrees_with_ground_truth(ground_truth, proba):
        return Variable(ground_truth) - proba < 0.5

    for i, output in enumerate(outputs):
        res.append([f(X_test[i]) > thresh])
        assert proba_agrees_with_ground_truth(f(X_test[i]) > thresh, output[0])
        #assert Variable(f(X_test[i]) > thresh) - output[0] < 0.5  # they agree


    assert binary_cross_entropy(logreg.predict(Matrix(X_test)), Matrix(res))[0][0] < 0.1
