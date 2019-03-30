import numpy as np


class Log:
    def __init__(self):
        self.f = lambda x: np.log(x)

    def __call__(self, x, y=None):
        return self.f(x)

    def get_grad(self, x, y):
        return (1 / x, None)
