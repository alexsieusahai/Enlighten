# Pax
A deep learning library written in Python with a NumPy backend that aims on being able to express its architecture as beautifully as possible.

## What's implemented so far?

* Automatic Differentiation
    * Currently, only Python Numbers are supported.

## What's on the immediate horizon?

* Automatic Differentiation
    * NumPy matrices, tensors in particular. I do _not_ want to use the PyTorch way of having to wrap everything in Tensors; I'd rather wrap parameters explicitly in Variables, which I'd obfuscate within a model by storing them in a parameters attribute on a layer by layer basis. I will lose speed for this, but I gain expressiveness.
* Model definition as function composition
    * `x = Variable(inp); params = xavier_init_normal((inp_shape, outp_shape)); b = np.zeros(outp_shape); f = W * x + b` is what I'm thinking; I do not want to have the user learn anything about this library to use it, other than wrapping _just one function_ into a Variable before running.
    * After, I can add model abstraction by defining synatic sugar for `feedforward`, for instance. 

## What needs to get done, but is not on the immediate horizon?
* Model definition as function composition
    * Recurrent neural networks will probably be constructed as an unrolled computational graph, where I will accumulate the gradients at each point as I need them. I might consider destroying the graph as I go on, once I have the gradients, in order to make it extremely memory efficient.
        * So, I'll need to have some kind of graph cleanup as I walk through the recursion stack; what I can do here is once I finish up the computations, I can delete the parents, since I'll never use them again.
	* Defining convolutions to fit well into my automatic differentiation paradigm.

## Usage examples

Demonstration of autograd to find point derivative of sigmoid (current path is in autograd folder):
```
import numpy as np

from variable import Variable

x = Variable(2)
f = 1 / (1 + np.e**(-1 * x))
print(f)
print(f.get_grad(x))
```
This will output:
```
0.8807970779778823
0.1049935854035065
```

We can do this for multiple arguments, and store into intermediate variables, as well:
```
import numpy as np

from variable import Variable

x = Variable(4)
z = Variable(2)
w = x * 3
a = w / z
print(a)
print(a.get_grad(x), a.get_grad(z))
```
This will output the following, as expected.
```
6.0
1.5 -3.0
```

This is an extremely expressive and beautiful way of doing automatic differentiation, in my opinion. This forms the backbone of the deep learning library.
