from .tree import Node

import sys
sys.path.append('.')
from loss_functions import binary_cross_entropy
from linear_models import LogisticRegression


class TreeClassifier(Node):
    """
    Wrapper around the Node class, using sensible default criterions and models.
    """
    def __init__(self, criterion: Callable=binary_cross_entropy, Model=LinearRegression, 
                 model_params: Dict={'alpha': 0.5, 'beta': 0.5}, num_splits: int=5, 
                 num_folds=5, min_samples=5):

        super().__init__(criterion, Model, model_params, num_splits, num_folds, min_samples)K
