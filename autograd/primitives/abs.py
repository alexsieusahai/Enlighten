class Abs:
    def __init__(self):
        self.f = lambda x: x if x.value >= 0 else -1 * x

    def __call__(self, x):
        return self.f(x)

    def get_grad(self, x):
        """
        Not the true gradient of abs (not defined at 0)
            but this should work anyways.
        """
        return 1 if x.value >= 0 else -1
