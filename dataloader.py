import sys
sys.path.append('.')
from autograd import Variable, Matrix

class DataLoader:

    def __init__(self, X: Matrix, y: Matrix=None):
        """
        :param X: The examples to iterate through.
        :param y: The outputs of the examples to iterate through. If None, then 
            this DataLoader is used for test data, and y_batch will not be returned in the  iterable.
        """
        self.X = X
        self.y = y
        self.i = 0
        if y is not None and len(X) != len(y):
            print('The lengths of X and y passed into DataLoader do not agree.')
            raise ValueError

    def get_output(self, i):
        return Matrix([self.X[i]]), Matrix([self.y[i]])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_output(idx)
        if isinstance(idx, list):
            X, y = [], []
            for i in idx:
                X.append(self.X[i])
                y.append(self.y[i])
            return DataLoader(Matrix(X), Matrix(y))

    def __iter__(self):
        while self.i < len(self.X):
            yield self.get_output(self.i)
            self.i += 1

    def __len__(self):
        return len(self.X)

    def peek(self):
        return self.get_output(self.i)
