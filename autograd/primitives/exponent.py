import numpy as np


class Exponent:
    def __init__(self):
        self.f = lambda x, y: x ** y
        
    def __call__(self, x, y):
        return self.f(x, y)
    
    def get_grad(self, x, y):
        return (y * x ** (y-1), x**y * np.log(x))
