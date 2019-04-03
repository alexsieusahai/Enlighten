class Divide:
    def __init__(self):
        pass 

    def __call__(self, f, g):
        return f / g
    
    def get_grad(self, f, f_prime, g, g_prime):
        if g == 0:
            return float('inf')
        return (g*f_prime - f*g_prime) / (g ** 2)
