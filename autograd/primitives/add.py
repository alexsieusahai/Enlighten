class Add:
    def __init__(self):
        pass
        
    def __call__(self, f, g):
        return f + g
    
    def get_grad(self, f, f_prime, g, g_prime):
        if f_prime is None or g_prime is None:
            return None
        return f_prime + g_prime
