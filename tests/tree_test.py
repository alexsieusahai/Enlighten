import numpy as np

from ..autograd import Variable, Matrix
from ..tree_models import Node, MajorityClassifier, negative_accuracy
from ..dataloader import DataLoader
from ..optimizers import SGD


def test_tree():
    num_cols, num_samples = 10, 10000
    X = []
    y = []

    for _ in range(num_samples):
        target = np.random.randint(0, 2)
        X.append([np.random.random() for _ in range(num_cols-1)] + [target])
        y.append([target])

    loader = DataLoader(X, y)

    tree = Node(negative_accuracy, MajorityClassifier, {}, num_splits=20, min_samples=10)

    for i in range(len(loader.y)):
        loader.y[i] = [loader.y[i][0] > 4]

    tree.fit(loader.X, loader.y)

    preds = tree.predict(loader.X)

    assert -1 * negative_accuracy(preds, loader.y) == 1
