# Pax
A deep learning library written in Python that aims on being able to express its architecture as beautifully as possible.  
Made almost exclusively for learning more about deep learning; no NumPy dependency, but very slow.

## What's implemented so far?

* Automatic Differentiation

## What's on the immediate horizon?

* Model definition as function composition
    * `x = Variable(inp); params = xavier_init_normal((inp_shape, outp_shape)); b = np.zeros(outp_shape); f = W * x + b` is what I'm thinking; I do not want to have the user learn anything about this library to use it, other than wrapping _just one function_ into a Variable before running.
    * After, I can add model abstraction by defining synatic sugar for `feedforward`, for instance. 

## What needs to get done, but is not on the immediate horizon?
* Model definition as function composition
    * Recurrent neural networks will probably be constructed as an unrolled computational graph, where I will accumulate the gradients at each point as I need them. I might consider destroying the graph as I go on, once I have the gradients, in order to make it extremely memory efficient.
        * So, I'll need to have some kind of graph cleanup as I walk through the recursion stack; what I can do here is once I finish up the computations, I can delete the parents, since I'll never use them again.
	* Defining convolutions to fit well into my automatic differentiation paradigm.

## Autograd
### How does it work?
For every single function, we can consider it as a composition of primitive functions, and by the chain rule and commutativity of multiplication, we can walk any way we'd like through the computational graph on an undirected path from any start variable to the output. This implementation of autograd considers only functions that takes in 2 arguments, and we consider a function of 3 arguments as a composition of 2 functions, for instance. 

What this implementation of autograd does is it starts at the output node, and it walks through the computational subgraph of one of its children, until it hits a node. Then, that node finally returns the identity function and the corresponding derivative (1), and then we can compute the derivative of a given node with respect to its immediate children, and finally by chain rule, we can obtain the derivative of the goal node with respect to any of the inputs. We do this by keeping track of each variable in a dictionary holding all of the gradients, where the keys are the locations of the Variables in memory. 

We store the value of each node, and (**not implemented yet, but it's on the horizon**) we delete the old intermediate nodes once we have no use for them anymore, sparing only the variables, so that they keep their locations in memory.
### Usage examples

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
