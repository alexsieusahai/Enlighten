from .generate_linear_dataset import generate_linear_dataset

from ..autograd import Variable, Matrix
from ..tree_models import Node, MajorityClassifier, negative_accuracy
from ..dataloader import DataLoader
from ..optimizers import SGD


def test_tree():

    tree = Node(negative_accuracy, MajorityClassifier, {}, num_splits=20, min_samples=10)
    num_cols, num_samples = 3, 10000
    loader, true_params = generate_linear_dataset(num_cols, num_samples)

    for i in range(len(loader.y)):
        loader.y[i] = [loader.y[i][0] > 4]

    tree.fit(loader.X, loader.y)

    preds = tree.predict(loader.X)

    assert -1 * negative_accuracy(preds, loader.y) > 0.5
