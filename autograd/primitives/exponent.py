import numpy as np


class Exponent:
    def __init__(self):
        pass
        
    def __call__(self, f, g):
        return g ** f
    
    def get_grad(self, f, f_prime, g, g_prime):
        if f_prime is None or g_prime is None:
            return None
        return g**f * np.log(g) * f_prime
