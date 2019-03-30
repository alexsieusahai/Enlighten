import sys
sys.path.append('..')
from autograd import Variable, Matrix

class DataLoader:

    def __init__(self, X: Matrix, y: Matrix=None, minibatch_size: int=1):
        """
        :param X: The examples to iterate through.
        :param y: The outputs of the examples to iterate through. If None, then 
            this DataLoader is used for test data, and y_batch will not be returned in the  iterable.
        :param minibatch_size: The size of the minibatch to use.
        """
        self.X = X
        self.y = y
        if y is not None and len(X) != len(y):
            print('The lengths of X and y passed into DataLoader do not agree.')
            raise ValueError
        self.minibatch_size = minibatch_size

    def __iter__(self):
        X_batch, y_batch = [], []
        for i in range(len(self.X)):
            X_batch.append(self.X[i])
            if self.y is not None:
                y_batch.append([self.y[i]])
            if len(X_batch) == self.minibatch_size:
                if self.y is not None:
                    yield Matrix(X_batch), Matrix(y_batch)
                else:
                    yield Matrix(X_batch)
                X_batch, y_batch = [], []
