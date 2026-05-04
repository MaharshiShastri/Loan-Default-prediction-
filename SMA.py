"""
This experiment explores the role of the perceptron algorithm as the fundamental building block of neural networks in deep learning.
A neural network is structured in layers, starting with an input layer, followed by one or more hidden layers, and ending with an output layer.
Each layer consists of multiple perceptrons that process incoming data by applying weights and passing the result through an activation function. 
The processed data flows layer by layer, enabling the network to transform raw input into meaningful predictions.

The experiment also highlights the learning process through backpropagation, where the difference between predicted and actual output is calculated as an error. 
This error is used to iteratively adjust the weights of the perceptrons, improving the model’s accuracy over time. 
Activation functions, such as sign, step, and sigmoid functions, play a crucial role in introducing non-linearity and determining whether a neuron activates, 
making the network capable of solving complex problems.
"""

# importing Python library 
import numpy as np
 
# define unit step function 
def unitStep(v): 
	if v >= 0: 
		return 1 
	else: 
		return 0 

# design Perceptron Model 
def perceptronModel(x, w, b): 
	v = np.dot(w, x) + b 
	y = unitStep(v) 
	return y 

# AND Logic Function 
# w1 = 1, w2 = 1, b = -1.5 
def AND_logicFunction(x): 
	w = np.array([1, 1]) 
	b = -1.5 
	return perceptronModel(x, w, b)
 
# testing the perceptron model 
test1 = np.array([0, 1]) 
test2 = np.array([1, 1]) 
test3 = np.array([0, 0]) 
test4 = np.array([1, 0]) 

print("AND({}, {}) = {}".format(0, 1, AND_logicFunction(test1))) 
print("AND({}, {}) = {}".format(1, 1, AND_logicFunction(test2))) 
print("AND({}, {}) = {}".format(0, 0, AND_logicFunction(test3))) 
print("AND({}, {}) = {}".format(1, 0, AND_logicFunction(test4)))
