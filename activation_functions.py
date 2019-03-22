import numpy as np

from autograd import Matrix

def sigmoid(x):
    f = lambda x: 1 / (1 + np.e**(-x))
    if isinstance(x, Matrix):
        return x.elementwise_apply(f)
    return f(x)

def relu(x):
    f = lambda x: x if x.value >= 0 else x*0
    if isinstance(x, Matrix):
        return x.elementwise_apply(f)
    return f(x)
