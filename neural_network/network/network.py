"""
=======
Network
=======

This module provie the machinary to build a basic neural
network using base functionality. This code is written
for learning purpose based on the code given on the
web-book: http://neuralnetworksanddeeplearning.com/chap1.html

Author:
The implementation is of the stochastical gradient decent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropogation. 

"""

from typing import List
from types import MethodType

import random

import numpy as np

class Network(object):
    def __init__(self, sizes: List[int]):
        """Creates a network with inital weights and biases
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        # The biases and weights here are intialized randomly using
        # numpy.random package's randn function that generates a matrix
        # of given dimensions with elements coming from a Guassian
        # distribution with mean 0 and standard deviation 1.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return Network.sigmoid(z)*(1 - Network.sigmoid(z))

    def feed_forward(self, a):
        """Return the output of the network if you a is an input        
        """
        for b, w in zip(self.biases, self.weights):
            a = Network.sigmoid(np.dot(w, a) + b)

        return a

    def stochastic_gradient_decent(
            self,
            training_data,
            epochs,
            mini_batch_size,
            eta,
            test_data=None,      
        ):
        """Trains the model using stochastic gradient decent

        The 'training_data' is a list tuples (x, y) representing
        training inputs and the desired outputs. The variable
        'epochs' is the number of epochs to train for. The
        'min_batch_size' is the size of the mini-batches to user
        when sampling. The 'eta' is the learning rate. The optional
        'test_data' is to enable partial progress.
        
        """
        # If test_data is there n_test will record its length
        if test_data:
            n_test = len(test_data)

        # Length of training_data is captured to use it when
        # creating mini_batches later
        n = len(training_data)

        # For each epoch the training is done with every
        # mini_batch of training data which are created
        # after the latter is randomly shuffeled.
        for j in range(epochs):
            # Randomly shuffle the training_data
            random.shuffle(training_data)
            
            # Partition the training data into minibatches
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # For each mini_batch following applies a single step of
            # gradient decent with given eta using the method
            # update_with_mini_batch
            for mini_batch in mini_batches:
                self.update_with_mini_batch(mini_batch, eta)
            
            # With testing data available the completion of epoch is
            # reported with the evaluation of the model on the
            # testing data normalized using the size of test data from
            # n_test variable. Otherwise, just the completion is reported.
            if test_data:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete.')


    def update_with_mini_batch(self, mini_batch, eta):
        """
        Updates the network weights and biases according to a single
        iteration of gradient decent, and using just the training data
        in `mini_batch`.

        `mini_batch` is a list of tuples `(x, y)`, and `eta` is the
        learning rate.
        """
        # The "nabla"b and "nabla"w are initalizaed here having zeros
        # in the same shape as elements of original biases and weights
        # matrices and arranged in the same sequence so tha they can
        # be added with them later.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # For each element (x, y) of the mini_batch, back propogation
        # is applied to it with the purpose of gaining the delta change
        # in the nabla_b and nabla_w due to the cost incurred by the
        # current network in predicting y with x. Later, these deltas
        # are used to update the neblas.
        for x, y in mini_batch:
            # The delta of biases and weights are computed from the observation
            # of y from a. This is done using back_propagation method.
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)   

            # The values of nabla_b(w) are updated using that of delta_nebla_b(w).
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # The biases are updated using b = b - eta/n*nb rule, which
        # is same as b = b - eta/n*sum(dnb) for n = len(mini_batches)
        self.biases = [
            b -(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)
        ]

        # The weights are updated using similar rule as biases.
        self.weights = [
            w -(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)
        ]


    def back_propagation(self, x, y):
        """
        Returns a tuple ``(nabla_a, nabla_b) representing the
        gradient for the cost function C_x. ``nabla_b`` and ``nabla_w`` are
        layer-by-layer lists of numpy arrays, similar to ``self.biases``
        and ``self.weights``. In fact, the similarity extends to each element
        in that they have same shapes and can be added to each other.
        
        """
        # The ``nabla_b`` and ``nabla_w are initialized element by element as
        # arrays of zeros of the same shape in the corresponding element of
        # the attributes ``biases`` and ``weights`` respectively.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feed forward
        activation = x
        
        # list to store the activations layer by layer
        activations = [activation]

        # list to store all the z vectors layer by layer
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b

            zs.append(z)

            activation = Network.sigmoid(z)

            activations.append(activation)

        ## Backward pass
        delta = (
            self.cost_derivative(activations[-1], y)
            * Network.sigmoid_prime(zs[-1])
        )

        # The nabla_b of the parameters in the outer arrays in
        # biases and weights are easiest to compute.
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # The hidden arrays going in from the output layer uses
        # the delta partials along with other partials generated
        # in between the layers using a form of chain rule.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = Network.sigmoid_prime(z)
            delta = (
                np.dot(self.weights[-(l-1)].transpose(), delta)
                * sp
            )
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-(l+1)].transpose())

        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        """
        Evaluates the number of test inputs for which the neural
        network ouputs the correct results. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer that has the highest activation.
        """

        test_results = [(np.argmax(self.feed_forward(x)), y) for x, y in test_data]

        return sum([a == y for a, y in test_results])

    def cost_derivative(self, output_activations, y):
        """
        Returns the vector of partial derivatives
        \partial C_x \partial a for the output activations.
        """
        # For C(a) = 1/2*(a - y) ^ 2, C'(a) = (a-y)
        return (output_activations - y)

