# Torchlite 
A deep learning library written in Python that aims on being able to express deep learning as beautifully as possible.  
Made almost exclusively for learning more about deep learning; very expressive, natural code, but (probably very) slow.

## What's implemented so far?

* Automatic Differentiation
* Naive matrix library
    * Somewhat cache friendly matrix multiplication since I access the matrices row by row rather than column by column. Seems to give a nice boost empirically.
    * Implementation of matrix transpose
    * Normal initialization via numpy
* Model definition as function composition
    * `f = sigmoid(W * x + b)` (a simple feedforward network) works; we can get the gradient of it just by doing `f.get_grad(W)`.
* Optimizers
	* Stochastic gradient descent
	* Adam
* Graph cleanup as I walk through the recursion stack.
    * After I pass a Variable which I know I will never use again, why keep it? Lets `del`ete it and move on.

## What's on the immediate horizon?

* Addition of minibatching.
* Synatic sugar for feedforward networks.
* `xor` example.
* Model definition as function composition.
    * Recurrent neural networks will probably be constructed as an unrolled computational graph, where I will accumulate the gradients at each point as I need them. I might consider destroying the graph as I go on, once I have the gradients, in order to make it extremely memory efficient.

## What needs to get done, but is not on the immediate horizon?
* Defining convolutions to fit well into my automatic differentiation paradigm.
* Usage examples solving common problems, such as MNIST.

## Autograd
### How does it work?
For every single function, we can consider it as a composition of primitive functions, and by the chain rule and commutativity of multiplication, we can walk any way we'd like through the computational graph on an undirected path from any start variable to the output. This implementation of autograd considers only functions that takes in 2 arguments, and we consider a function of 3 arguments as a composition of 2 functions, for instance. 

What this implementation of autograd does is it starts at the output node, and it walks through the computational subgraph of one of its children, until it hits a node. Then, that node finally returns the identity function and the corresponding derivative (1), and then we can compute the derivative of a given node with respect to its immediate children, and finally by chain rule, we can obtain the derivative of the goal node with respect to any of the inputs. We do this by keeping track of each variable in a dictionary holding all of the gradients, where the keys are the locations of the Variables in memory. 

We store the value of each node, and we delete the old intermediate nodes once we have no use for them anymore, sparing only the variables, so that they keep their locations in memory.
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

## Naive Matrix Library
### Why do this?
If I drop down to NumPy, it's impossible for me to use Autograd; I suspect this is why in PyTorch that everything must be wrapped in its Tensor types.
