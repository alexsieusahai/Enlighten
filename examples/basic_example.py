if __name__ == "__main__":
    from activation_functions import sigmoid
    from autograd import Variable, Matrix, zeros

    W0 = zeros(2, 2)
    W0.init_normal()

    b0 = zeros(2, 1)
    b0.init_normal()

    W1 = zeros(1, 2)
    W1.init_normal()

    b1 = zeros(1, 1)
    b1.init_normal()

    x = Matrix([
            [Variable(1)],
            [Variable(2)]
        ])

    hidden = sigmoid(W0 * x + b0)
    output = sigmoid(W1 * hidden + b1)
    print(output)
    print(output.get_grad(W0))
    print(output.get_grad(b0))
    print(output.get_grad(W1))
    print(output.get_grad(b1))
