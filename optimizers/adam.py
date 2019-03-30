try:
    from .optimizer import Optimizer
except ImportError:
    from optimizer import Optimizer


class Adam(Optimizer):
    """
    See Algorithm 1 in https://arxiv.org/pdf/1412.6980.pdf.
    """
    def __init__(self, alpha, beta1, beta2, eps=10e-8, minibatch_size: int=1):
        super().__init__(minibatch_size)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_val = eps
        self.eps_dict = {}
        self.t_dict = {}
        self.m_dict = {} 
        self.v_dict = {} 
        self.reset_minibatch_grad()

    def store_params(self, t, m, v, eps, new_id):
        self.t_dict[new_id] = t
        self.m_dict[new_id] = m
        self.v_dict[new_id] = v
        self.eps_dict[new_id] = eps

    def delete_params(self, old_id):
        del self.t_dict[old_id]
        del self.m_dict[old_id]
        del self.v_dict[old_id]
        del self.eps_dict[old_id]

    def step(self, x, x_grad):
        self.accumulate_minibatch_grad(x_grad)
        if self.minibatch_accumulated():
            if id(x) not in self.t_dict:
                self.t_dict[id(x)] = 0
                self.m_dict[id(x)] = x_grad.zeros()
                self.v_dict[id(x)] = x_grad.zeros()
                self.eps_dict[id(x)] = x_grad.initialize_matrix(self.eps_val)

            self.t_dict[id(x)] += 1

            t, m, v, eps = self.t_dict[id(x)], self.m_dict[id(x)], self.v_dict[id(x)], self.eps_dict[id(x)]

            m = self.beta1 * m + (1-self.beta1) * x_grad
            v = self.beta2 * v + (1-self.beta2) * x_grad.elementwise_apply(lambda x: x**2)
            curr_alpha = self.alpha * ((1 - self.beta2**t)**0.5) / (1-self.beta1**t)
            rescaled_params = x - curr_alpha * m.hadamard((v.elementwise_apply(lambda x: x**0.5) + eps).elementwise_apply(lambda x: 1 / x))
            rescaled_params = rescaled_params.reset_grad()

            self.store_params(t, m, v, eps, id(rescaled_params))
            self.delete_params(id(x))

            self.reset_minibatch_grad()
            return rescaled_params

        return x
        


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    sys.path.append('../autograd')
    from autograd import Variable, zeros, Matrix
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
    print(f'Final output is {output}')
