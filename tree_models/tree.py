from itertools import product
from typing import Callable, Dict

import numpy as np

import sys
sys.path.append('.')
from autograd import Variable, Matrix
from dataloader import DataLoader
from validation import KFold


def split(X, y, split_val, col):
    """
    Splits X into two parts; one where X[col] < split_val for every row, and one where
    X[col] >= split_val for every row.
    """
    X_l, X_r = [], []
    y_l, y_r = [], []
    for i, row in enumerate(X):
        if row[col] < split_val:
            X_l.append(row)
            y_l.append(y[i])
        else:
            X_r.append(row)
            y_r.append(y[i])
    X_l, X_r = Matrix(X_l), Matrix(X_r)
    y_l, y_r = Matrix(y_l), Matrix(y_r)
    return X_l, y_l, X_r, y_r

class Node:
    def __init__(self, criterion: Callable, Model, model_params: Dict, 
                 num_splits: int=5, num_folds=5, min_samples=5):
        """
        :param criterion: The splitting criterion to use.
        :param Model: The Model initalizer which will be used to measure the "goodness" of the 
            criterion at each split, and also used to make predictions at the leaf node.
        :param model_params: The parameters to pass in to the model.
        :param num_splits: The number of splits to make at any column.
            In this scenario, we randomly select a value num_splits times, assuming a uniform
                prior over each feature. Uses np.random.uniform to do this, so you can reproduce 
                results for multiple trials using np.random.seed.
        :param num_folds: The number of folds to use to evaluate the split.
        :param min_samples: The minimum samples that must be in a leaf. Must be at least num_folds.
        """
        self.criterion = criterion
        self.Model = Model
        self.model = None
        self.model_params = model_params
        self.num_splits = num_splits
        self.num_folds = num_folds
        self.min_samples = min_samples

    def _get_oof_score(self, X: Matrix, y: Matrix):
        """
        Gets the average criterion output for the model passed in on 
            the out of fold values, using a KFold(num_folds) cv.
        """
        model = self.Model(**self.model_params)
        avg_oof = 0
        loader = DataLoader(X, y)
        for train, valid in KFold(self.num_folds).split(loader):
            model.fit(train)
            avg_oof += self.criterion(model.predict(valid.X), valid.y) / self.num_folds
        return avg_oof


    def fit(self, X: Matrix, y: Matrix):
        """
        Fits the model to the data.
        The splits aim to minimize the criterion.
        """
        if len(X) == 0:
            assert False
        best_oof = self._get_oof_score(X, y)
        self.best_split = None

        split_inds = [np.random.randint(0, len(X)) for _ in range(self.num_splits)]
        for split_ind in split_inds:
            for col in range(len(X[0])):
                split_val = X[split_ind][col]

                X_l, y_l, X_r, y_r = split(X, y, split_val, col)
                if len(X_l) < self.min_samples or len(X_r) < self.min_samples:
                    continue

                oof_l = self._get_oof_score(X_l, y_l)
                oof_r = self._get_oof_score(X_r, y_r)
                oof = (oof_l * len(X_l) + oof_r * len(X_r)) / len(X)
                if oof < best_oof:
                    best_oof = oof
                    self.best_split = (split_val, col)

        if self.best_split is None:
            self.model = self.Model(**self.model_params)
            loader = DataLoader(X, y)
            if len(X) == 0:
                assert False
            if len(loader) == 0:
                assert False
            self.model.fit(loader)
        else:
            split_val, col = self.best_split
            X_l, y_l, X_r, y_r = split(X, y, split_val, col)

            self.left_child = Node(self.criterion, self.Model, self.model_params,
                    self.num_splits, self.num_folds, self.min_samples)
            self.left_child.fit(X_l, y_l)

            self.right_child = Node(self.criterion, self.Model, self.model_params,
                    self.num_splits, self.num_folds, self.min_samples)
            self.right_child.fit(X_r, y_r)

    def predict_for_row(self, row: Matrix):
        if self.best_split is None:
            return self.model.predict([row])
        else:
            split_val, split_col = self.best_split
            go_left = row[split_col] < split_val

            if go_left:
                return self.left_child.predict_for_row(row)
            else:
                return self.right_child.predict_for_row(row)

    def predict(self, X: Matrix):
        preds = []
        for row in X:
            preds.append(self.predict_for_row(row)[0])
        return Matrix(preds)


if __name__ == "__main__":
    from majority_classifier import MajorityClassifier
    from metrics import negative_accuracy

    criterion = negative_accuracy 
    Model = MajorityClassifier

    tree = Node(criterion, Model, {})

    num_samples = 1000
    X, y = [], []
    for _ in range(num_samples):
        target = np.random.randint(0, 2)
        X.append([np.random.random(), target])
        y.append(target)

    X, y = Matrix(X), Matrix(y)
    
    tree.fit(X, y)

    preds = (tree.predict(X))
    print(negative_accuracy(preds, y))
