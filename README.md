# Enlighten
Most of supervised learning seems to be about postulating that the dataset's output fits to some kind of manifold with respect to its inputs, modelling that manifold using some variables, then optimizing either the likelihood of the data given the model or vice versa. The driving horse seems to be differentiation, with a couple notable exceptions (a good one is SVMs being optimized by quadratic programming techniques).  
How far can we push autodiff? Can we make a fully functioning (albeit, _really_ slow) very customizable general supervised learning library using (mostly) just autodiff and synatic sugar?

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
* LinearRegression object, and the basic interface for models
* DataLoader object (handles minibatching)
* Common loss functions (MSE, MAE, CrossEntropy)
* Logistic regression
* Decision Tree Regression / Classification
* Multicore processing for matrix multiplication
    * For large matrices, it is used. For smaller matrices, the overhead of instantiating processes is not worth it (I avoided serializing and deserializing big objects which gave a very big speed boost on large matrices).

## What's on the immediate horizon?

* Add Random Forest to tree based models using implemented decision tree
* Synatic sugar for feedforward networks
* `xor` example
* Model definition as function composition
    * Recurrent neural networks will probably be constructed as an unrolled computational graph, where I will accumulate the gradients at each point as I need them

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

Multiple variables being used over and over is supported, as well, using basic derivative rules of calculus:
```
from variable import Variable

x = Variable(4)
z = Variable(2)
f = x\*\*2 + (x + z) / (x + 5 * z)
print(f)
print(f.get\_grad(x), f.get\_grad(z))
```
This will output the following, as expected.
```
16.428571428571427
8.040816326530612 -0.08163265306122448
```

## Naive Matrix Library
### Why do this?
If I drop down to NumPy, it's impossible for me to use Autograd; I suspect this is why in PyTorch that everything must be wrapped in its Tensor types. This makes everything ridiculously slow, but I'm not sure about an easy workaround at the moment. For the toy problems that this library will be used on, this will be fine.
