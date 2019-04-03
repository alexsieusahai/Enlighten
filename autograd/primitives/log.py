import numpy as np


class Log:
    def __init__(self):
        pass

    def __call__(self, f, g=None):
        return np.log(f)

    def get_grad(self, f, f_prime, g=None, g_prime=None):
        if f_prime is None:
            return None
        return f_prime / f
