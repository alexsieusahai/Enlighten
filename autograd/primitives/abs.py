class Abs:
    def __init__(self):
        pass

    def __call__(self, f, y=None):
        return f if f >= 0 else -1 * f

    def get_grad(self, f, f_prime, g=None, g_prime=None):
        """
        Not the true gradient of abs (not defined at 0)
            but this should work anyways.
        """
        if f_prime is None:
            return None
        return f_prime if f >= 0 else -f_prime
