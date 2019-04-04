import sys
sys.path.append('.')
from autograd import Variable, Matrix
from dataloader import DataLoader


class MajorityClassifier:
    """
    Classifies using majority classification.
    """
    def __init__(self):
        pass

    def fit(self, loader: DataLoader):
        outputs = {}
        max_output = -1
        max_output_class = None
        for (_, output) in loader:
            outputs[output] = outputs.get(output, 0) + 1
            if outputs[output] > max_output:
                max_output = outputs[output]
                max_output_class = output

        self.pred = max_output_class[0]

    def predict(self, X: Matrix):
        return Matrix([self.pred for _ in X])
