from autograd import Variable, Matrix


def mean_squared_error(x, y):
    return ((x - y)**2).mean()

def mean_absolute_error(x, y):
    return (x - y).abs().mean()

def binary_cross_entropy(pred, target):
    """
    This assumes all classes are either 0 or 1.
    If more than 1 sample is passed in, the mean is taken.
    """
    ones = pred.initialize_matrix(1)
    return -1 * (target.hadamard(pred.log()) + (ones-target).hadamard((ones - pred).log())).mean()

if __name__ == "__main__":
    x = Matrix([
        [Variable(0.1)],
        [Variable(0.3)],
        [Variable(0.6)]
        ])

    y = Matrix([
        [Variable(0)],
        [Variable(1)],
        [Variable(0)]
        ])

    print(mean_squared_error(x, y))
    print(mean_absolute_error(x, y))
    print(binary_cross_entropy(x, y))
