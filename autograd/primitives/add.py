class Add:
    def __init__(self):
        self.f = lambda x, y: x + y
        
    def __call__(self, x, y):
        return self.f(x, y)
    
    def get_grad(self, x, y):
        return (1, 1)
