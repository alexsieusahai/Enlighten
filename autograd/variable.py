try:
    from .primitives import Add, Multiply, Divide, Exponent, Abs
except ImportError:
    from primitives import Add, Multiply, Divide, Exponent, Abs

class Variable:
    def __init__(self, parent0, parent1=None, primitive=None, eager=True):
        is_variable = parent1 is None
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
        node = self.ensure_node(node)
        return Variable(self, node, Add())
    
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
        return Variable(self, node, Exponent())
    
    def __rpow__(self, node):
        return Variable(node, self, Exponent())

    def abs(self):
        return Variable(self, None, Abs())
        
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
        parent0_grad, parent1_grad = self.primitive.get_grad(self.parent0.compute(), self.parent1.compute())
        self.grad_dict = {}
        for key in self.parent0.grad_dict:
            self.grad_dict[key] = self.parent0.grad_dict[key] * parent0_grad
        for key in self.parent1.grad_dict:
            self.grad_dict[key] = self.parent1.grad_dict[key] * parent1_grad
            
    def get_grad(self, var):
        """
        Returns the gradient of Variable with respect to variable, using the 
            position of the variable passed in in memory.
        """
        return self.grad_dict[id(var)]
