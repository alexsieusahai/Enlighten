class Abs:
    def __init__(self):
        self.f = lambda x: x if x >= 0 else -1 * x

    def __call__(self, x, y=None):
        return self.f(x)

    def get_grad(self, x, y):
        """
        Not the true gradient of abs (not defined at 0)
            but this should work anyways.
        """
        return (1 if x >= 0 else -1, None)
