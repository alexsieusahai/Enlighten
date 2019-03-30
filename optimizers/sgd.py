class SGD:
    """
    Implementation of SGD with momentum.
    """
    def __init__(self, alpha: int, beta: int=0, minibatch_size: int=1):
        self.alpha = alpha
        self.beta = beta
        self.last_scaled_grad_dict = {}
        self.minibatch_size = minibatch_size
        self.reset_minibatch_grad()

    def reset_minibatch_grad(self):
        self.curr_size = 0
        self.avg_x_grad = None

    def accumulate_minibatch_grad(self, x_grad):
        self.curr_size += 1
        if self.avg_x_grad is None:
            self.avg_x_grad = x_grad
        else:
            self.avg_x_grad += x_grad / self.minibatch_size

    def minibatch_accumulated(self):
        return self.curr_size == self.minibatch_size

    def step(self, x, x_grad):
        self.accumulate_minibatch_grad(x_grad)
        if self.minibatch_accumulated():
            if id(x) not in self.last_scaled_grad_dict:
                self.last_scaled_grad_dict[id(x)] = x_grad

            scaled_grad = self.beta * self.last_scaled_grad_dict[id(x)] + self.alpha * self.avg_x_grad
            rescaled_params = x - scaled_grad
            rescaled_params = rescaled_params.reset_grad()
            del self.last_scaled_grad_dict[id(x)]
            self.last_scaled_grad_dict[id(rescaled_params)] = scaled_grad

            self.reset_minibatch_grad()
            return rescaled_params

        return x


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
