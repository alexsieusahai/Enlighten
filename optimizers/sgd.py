class SGD:
    """
    Implementation of SGD with momentum.
    """
    def __init__(self, alpha: int, beta: int=0):
        self.alpha = alpha
        self.beta = beta
        self.last_scaled_grad = None

    def step(self, x, x_grad):
        if self.last_scaled_grad is None:
            self.last_scaled_grad = x_grad

        scaled_grad = self.beta * self.last_scaled_grad + self.alpha * x_grad
        self.last_scaled_grad = scaled_grad
        return x - scaled_grad


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
    optimizer = SGD(0.10)

    target = zeros(1, 1)
    target[0][0] = 0.5
    for _ in range(1000):
        f = sigmoid(W*x + b)
        output = (f - target).abs()
        W = optimizer.step(W, output.get_grad(W))
        W = W.reset_grad()
    print(f)
    print(output)
