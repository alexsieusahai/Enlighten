from .generate_linear_dataset import generate_linear_dataset

from ..autograd import Variable, Matrix
from ..linear_models import LinearRegression
from ..optimizers import SGD
from ..validation import KFold


def test_linreg():
    num_cols, num_samples = 3, 10000
    loader, true_params = generate_linear_dataset(num_cols, num_samples)
    kfold = KFold(2)

    for train, valid in kfold.split(loader):
        linreg = LinearRegression()
        optim = SGD(0.1, 0.1, minibatch_size=1)
        linreg.fit(train, optim)

        eps = 1e-4

        for i in range(num_cols):
            assert (linreg.params[i][0] - true_params[i]).abs() < eps

        assert (linreg.bias[0][0] - 3).abs() < eps

        outputs = linreg.predict(Matrix(valid.X))
        y_valid = valid.y
        for i, output in enumerate(outputs):
            assert (output[0] - y_valid[i][0]).abs() < eps
