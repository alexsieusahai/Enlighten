from autograd import Variable, Matrix


def mean_squared_error(x, y):
    return ((x - y)**2).mean()

def mean_absolute_error(x, y):
    return (x - y).abs().mean()

def cross_entropy(x, y):
    """
    Note that x and y must be the same shape.
    """
    ones = x.initialize_matrix(1)
    return -1 * (y.hadamard(x.log()) + (ones-y).hadamard((ones-x).log())).sum()

if __name__ == "__main__":
    x = Matrix([
        [Variable(0.1)],
        [Variable(0.3)],
        [Variable(0.6)]
        ])

    y = Matrix([
        [Variable(0.5)],
        [Variable(0.1)],
        [Variable(0.4)]
        ])

    x - y

    print(mean_squared_error(x, y))
    print(mean_absolute_error(x, y))
    print(cross_entropy(x, y))
