try:
    from .primitives import Add, Multiply, Divide, Pow, Abs, Log, Exponent
except ImportError:
    from primitives import Add, Multiply, Divide, Pow, Abs, Log, Exponent

class Variable: 
    def __init__(self, parent0, parent1=None, primitive=None, eager=True):
        is_variable = primitive is None
        if not is_variable:
            self.parent0 = self.ensure_node(parent0)
            self.parent1 = self.ensure_node(parent1)
        self.primitive = primitive
        self.value = parent0 if is_variable else None
        self.grad_dict = {id(self): 1}
        if eager:
            self.value = self.compute()

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)
    
    def __neg__(self):
        return Variable(self, -1, Multiply())
        
    def __add__(self, node):
        return Variable(self, node, Add())

    def __sub__(self, node):
        node = node * -1
        return self + node
    
    def __radd__(self, node):
        return Variable(node, self, Add())
    
    def __mul__(self, node):
        return Variable(self, node, Multiply())
    
    def __rmul__(self, node):
        return Variable(node, self, Multiply())
    
    def __truediv__(self, node):
        return Variable(self, node, Divide())
    
    def __rtruediv__(self, node):
        return Variable(node, self, Divide())
    
    def __pow__(self, node):
        return Variable(self, node, Pow())

    def __rpow__(self, node):
        return Variable(self, node, Exponent())

    def __lt__(self, node):
        val = node.value if isinstance(node, Variable) else node
        return self.value < node

    def __le__(self, node):
        val = node.value if isinstance(node, Variable) else node
        return self.value <= node

    def __gt__(self, node):
        val = node.value if isinstance(node, Variable) else node
        return self.value > node

    def __ge__(self, node):
        val = node.value if isinstance(node, Variable) else node
        return self.value >= node

    def __eq__(self, node):
        val = node.value if isinstance(node, Variable) else node
        return self.value == node
    
    def __ne__(self, node):
        val = node.value if isinstance(node, Variable) else node
        return self.value == node

    def abs(self):
        return Variable(self, 'a', Abs())

    def log(self):
        return Variable(self, 'a', Log())
        
    def ensure_node(self, node):
        if isinstance(node, Variable):
            return node
        return Variable(node)
    
    def compute(self):
        if self.value is None:
            self.value = self.primitive(self.parent0.compute(), self.parent1.compute())
            self.compute_gradient()
        return self.value
    
    def compute_gradient(self):
        """
        Computes the gradient with respect to parent0, parent1.
        """
        self.grad_dict = {}
        parent0_val, parent1_val = self.parent0.compute(), self.parent1.compute()

        def update_grad(grad_dict):
            for key in grad_dict:
                grad = self.primitive.get_grad(parent0_val, self.parent0.grad_dict.get(key, 0), parent1_val, self.parent1.grad_dict.get(key, 0))
                if grad_dict[key] is not None:
                    self.grad_dict[key] = grad 
        
        update_grad(self.parent0.grad_dict)
        update_grad(self.parent1.grad_dict)

        if self.parent0.primitive is not None:
            del self.parent0
        if self.parent1.primitive is not None:
            del self.parent1
            
    def get_grad(self, var):
        """
        Returns the gradient of Variable with respect to variable, using the 
            position of the variable passed in in memory.
        """
        return self.grad_dict[id(var)]

    def reset_grad(self):
        self.grad_dict = {id(self): 1}
        return self


if __name__ == "__main__":
    import numpy as np

    x = Variable(3)

    f = x/x
    assert f.get_grad(x) == 0

    f = x*x*x*x
    assert f.get_grad(x) == 4*3**3

    y = Variable(5)
    f = (x*y) / (x + y)

    assert f.get_grad(x) == 25 / 64
    assert f.get_grad(y) == 9 / 64

    f = (-(x*y) / (x + y)).abs()

    assert f.get_grad(x) == 25 / 64
    assert f.get_grad(y) == 9 / 64

    f = x**2 * y
    assert f.get_grad(x) == 30
    assert f.get_grad(y) == 9

    f = (x**2).log()
    assert f.get_grad(x) == 2 / 3

    f = 1 / (1 + np.e**(-1 * (3*x + 5)))
    assert abs(f.get_grad(x) - 2.494582e-6) < 0.0001
