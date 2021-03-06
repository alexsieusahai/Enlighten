from ..validation import KFold
from ..autograd import Matrix
from ..dataloader import DataLoader


def test_KFold():
    kfold = KFold(5)

    X = Matrix([[i] for i in range(10)])
    y = Matrix([[i % 2] for i in range(10)])
    loader = DataLoader(X, y)

    for train_loader, valid_loader in kfold.split(loader):
        assert len(train_loader) in [8, 9]
        assert len(valid_loader) in [1, 2]
