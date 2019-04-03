import numpy as np


class Pow:
    def __init__(self):
        pass
        
    def __call__(self, f, g):
        return f ** g
    
    def get_grad(self, f, f_prime, c, c_prime):
        """
        Just power rule and chain rule
        """
        return c*f**(c-1) * f_prime
