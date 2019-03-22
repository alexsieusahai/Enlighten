if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from autograd import Variable, Matrix, zeros
    from activation_functions import sigmoid

    W0 = zeros(200, 2)
    W0.init_normal()

    b0 = zeros(200, 1)
    b0.init_normal()

    W1 = zeros(1, 200)
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
