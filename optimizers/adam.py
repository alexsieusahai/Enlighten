class Adam:
    """
    See Algorithm 1 in https://arxiv.org/pdf/1412.6980.pdf.
    """
    def __init__(self, alpha, beta1, beta2, eps=10e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_val = eps
        self.eps = None
        self.m = None
        self.v = None
        self.t = 0

    def step(self, x, x_grad):
        self.t += 1
        if self.m is None:
            self.m = x_grad.zeros()
            self.v = x_grad.zeros()
            self.eps = x_grad.initialize_matrix(self.eps_val)

        self.m = self.beta1 * self.m + (1-self.beta1) * x_grad
        self.v = self.beta2 * self.v + (1-self.beta2) * x_grad.elementwise_apply(lambda x: x**2)
        curr_alpha = self.alpha * ((1 - self.beta2**self.t)**0.5) / (1-self.beta1**self.t)
        return x - curr_alpha * self.m.hadamard((self.v.elementwise_apply(lambda x: x**0.5) + self.eps).elementwise_apply(lambda x: 1 / x))


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
    optimizer = Adam(0.001, 0.9, 0.999)

    target = zeros(1, 1)
    target[0][0] = 0.5
    for _ in range(1000):
        f = sigmoid(W*x + b)
        output = (f - target).abs()
        if not (_ % 100):
            print(output)
        W = optimizer.step(W, output.get_grad(W))
        W = W.reset_grad()
    print(f'Final output is {output}')
