import random

from ..dataloader import DataLoader


def generate_linear_dataset(num_cols, num_samples):
    true_params = [i for i in range(num_cols)]
    f = lambda row: sum([row[i] * true_params[i] for i in range(len(row))]) + 3
    X = []
    y = []
    for _ in range(num_samples): 
        X.append([random.random() for _ in range(num_cols)])
        y.append([f(X[-1])])

    return DataLoader(X, y), true_params
