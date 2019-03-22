class Adam:
    """
    See Algorithm 1 in https://arxiv.org/pdf/1412.6980.pdf.
    """
    def __init__(self, alpha, beta1, beta2, eps=10e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_val = eps
        self.eps_dict = {}
        self.t_dict = {}
        self.m_dict = {} 
        self.v_dict = {} 

    def step(self, x, x_grad):
        if id(x) not in self.t_dict:
            self.t_dict[id(x)] = 0
            self.m_dict[id(x)] = x_grad.zeros()
            self.v_dict[id(x)] = x_grad.zeros()
            self.eps_dict[id(x)] = x_grad.initialize_matrix(self.eps_val)

        self.t_dict[id(x)] += 1

        self.m_dict[id(x)] = self.beta1 * self.m_dict[id(x)] + (1-self.beta1) * x_grad
        self.v_dict[id(x)] = self.beta2 * self.v_dict[id(x)] + (1-self.beta2) * x_grad.elementwise_apply(lambda x: x**2)
        curr_alpha = self.alpha * ((1 - self.beta2**self.t_dict[id(x)])**0.5) / (1-self.beta1**self.t_dict[id(x)])
        rescaled_params = x - curr_alpha * self.m_dict[id(x)].hadamard((self.v_dict[id(x)].elementwise_apply(lambda x: x**0.5) + self.eps_dict[id(x)]).elementwise_apply(lambda x: 1 / x))
        self.t_dict[id(rescaled_params)] = self.t_dict[id(x)]
        self.m_dict[id(rescaled_params)] = self.m_dict[id(x)]
        self.v_dict[id(rescaled_params)] = self.v_dict[id(x)]
        self.eps_dict[id(rescaled_params)] = self.eps_dict[id(x)]
        del self.t_dict[id(x)]
        del self.m_dict[id(x)]
        del self.v_dict[id(x)]
        del self.eps_dict[id(x)]

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
    optimizer = Adam(0.001, 0.9, 0.999)

    target = zeros(1, 1)
    target[0][0] = 0.5
    for _ in range(5000):
        f = sigmoid(W*x + b)
        output = (f - target).abs()
        if not (_ % 100):
            print(output)
        W = optimizer.step(W, output.get_grad(W))
        W = W.reset_grad()
    print(f'Final output is {output}')
