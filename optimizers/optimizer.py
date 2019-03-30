class Optimizer:
    """
    Base class for all Optimizer objects.
    Implements minibatch handling logic, and sets the interface for all optimizers.
    """
    def __init__(self, minibatch_size: int=1):
        """
        :param minibatch_size: The size of the minibatch to use.
        """
        self.minibatch_size = minibatch_size

    def reset_minibatch_grad(self):
        self.curr_size = 0
        self.avg_x_grad = None

    def minibatch_accumulated(self):
        return self.curr_size == self.minibatch_size

    def accumulate_minibatch_grad(self, x_grad):
        self.curr_size += 1
        if self.avg_x_grad is None:
            self.avg_x_grad = x_grad
        else:
            self.avg_x_grad += x_grad / self.minibatch_size
