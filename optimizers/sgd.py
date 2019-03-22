class SGD:
    """
    Implementation of SGD with momentum.
    """
    def __init__(self, alpha: int, beta: int=0):
        self.alpha = alpha
        self.beta = beta
        self.last_scaled_grad_dict = {}

    def step(self, parameter_iterable, output):
        if id(x) not in self.last_scaled_grad_dict:
            self.last_scaled_grad_dict[id(x)] = x_grad

        scaled_grad = self.beta * self.last_scaled_grad_dict[id(x)] + self.alpha * x_grad
        rescaled_params = x - scaled_grad
        rescaled_params = rescaled_params.reset_grad()
        del self.last_scaled_grad_dict[id(x)]
        self.last_scaled_grad_dict[id(rescaled_params)] = scaled_grad
        return rescaled_params


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    sys.path.append('../autograd')
    from autograd import Variable, zeros
    from activation_functions import sigmoid

    W = zeros(1, 2)
    W.init_normal()
    b = zeros(1, 1)
    b.init_normal()

    x = zeros(2, 1)
    x.init_normal()  # so x is nonzero
    optimizer = SGD(0.1)

    target = zeros(1, 1)
    target[0][0] = 0.5
    for _ in range(5000):
        f = sigmoid(W*x + b)
        output = (f - target).abs()
        W = optimizer.step(W, output.get_grad(W))
        b = optimizer.step(b, output.get_grad(b))
    print(f)
    print(output)
