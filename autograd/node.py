from variable import Variable
from primitives import Add, Multiply, Divide, Exponent


class Node:
    def __init__(self, parent0, parent1, primitive, eager=True):
        self.parent0 = self.ensure_node(parent0)
        self.parent1 = self.ensure_node(parent1)
        self.primitive = primitive
        self.value = None
        if eager:
            self.value = self.compute()
            
    def __str__(self):
        return str(self.value)
        
    def ensure_node(self, node):
        if isinstance(node, Node) or isinstance(node, Variable):
            return node
        return Variable(node)
    
    def __neg__(self):
        return Node(self, -1, Multiply())
        
    def __add__(self, node):
        node = self.ensure_node(node)
        return Node(self, node, Add())
    
    def __radd__(self, node):
        return Node(node, self, Add())
    
    def __mul__(self, node):
        return Node(self, node, Multiply())
    
    def __rmul__(self, node):
        return Node(node, self, Multiply())
    
    def __truediv__(self, node):
        return Node(self, node, Divide())
    
    def __rtruediv__(self, node):
        return Node(node, self, Divide())
    
    def __pow__(self, node):
        return Node(self, node, Exponent())
    
    def __rpow__(self, node):
        return Node(node, self, Exponent())
    
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
            
    def get_grad(self, var: Variable):
        """
        Returns the gradient of Node with respect to variable, using the 
            position of the variable passed in in memory.
        """
        return self.grad_dict[id(var)]
