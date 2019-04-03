class Multiply:
    def __init__(self):
        pass
        
    def __call__(self, f, g):
        return f*g
    
    def get_grad(self, f, f_prime, g, g_prime):
        return f_prime * g + f * g_prime
