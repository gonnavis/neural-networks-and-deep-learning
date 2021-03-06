"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import pdb

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes # [784, 30, 10]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        ''' biases len(2)
          [
            array(30, 1),
            array(10, 1),
          ]
        '''
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        pdb.set_trace()
        ''' zip(sizes[:-1], sizes[1:])
                                    [(784,30),(30,10)]
        '''
        ''' weights len(2)
          [
            array(30, 784),
            array(10, 30),
          ]
        '''

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
          '''
                    [
                      (
                        array([
                          [x],
                          ...30
                        ]),
                        array([
                          [x,...784],
                          ...30
                        ]),
                      ),
                      (
                        array([
                          [x],
                          ...10
                        ]),
                        array([
                          [x,...30],
                          ...10
                        ]),
                      )
                    ]
          '''

          # pdb.set_trace() 
          # (Pdb) p b
          # (Pdb) p w
          # (Pdb) c

            # print 'w------------------------------------------------------------------'
            # print w
            # print 'a------------------------------------------------------------------'
            # print a
            # print 'b------------------------------------------------------------------'
            # print b
            # raw_input()

          a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data) # 50000

        # print '+++++++++++++++++++++++++++++++'
        # print 'n'
        # print n
        # print '-------------------------------'
        # raw_input()

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
    
            # # mini_batches.txt
            # print 'mini_batches<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
            # print len(mini_batches) # 5000
            # print len(mini_batches[0]) # 10
            # print len(mini_batches[0][0]) # 2
            # print len(mini_batches[0][1]) # 2
            # print 'mini_batches>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
            # raw_input()

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
    
                # # mini_batch.txt
                # print 'mini_batch<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
                # print len(mini_batch) # 10
                # print len(mini_batch[0]) # 2
                # print len(mini_batch[0][0]) # 784
                # print len(mini_batch[0][1]) # 10
                # print     mini_batch[0][1] # [[ 0.] [ 0.] [ 0.] [ 0.] [ 0.] [ 1.] [ 0.] [ 0.] [ 0.] [ 0.]]
                # print 'mini_batch>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                # raw_input()

            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test) # Epoch 0: 9054 / 10000
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # print '+++++++++++++++++++++++++++++++'
        # # print len(mini_batch[0])
        # print len(mini_batch[0][0]) # 784
        # # print mini_batch[0][0]
        # print '-------------------------------'
        # raw_input()

        # print len(mini_batch[0][0]) # 784

        # len(mini_batch) // 10
        # mini_batch[0] // tuple
        # mini_batch[0][0].shape // (784, 1)
        # mini_batch[0][1].shape // (10, 1)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        '''
        >>> len(net.weights)  // 2
        >>> len(net.weights[0])  // 30
        >>> len(net.weights[1])  // 10
        >>> len(net.weights[0][0])  // 784
        >>> len(net.weights[0][1])  // 784
        >>> len(net.weights[0][2])  // 784
        >>> len(net.weights[1][0])  // 30
        >>> len(net.weights[1][1])  // 30
        >>> len(net.weights[0][0][0])  // error

        weights[
            [ // 30
            ],
            [ // 10
            ].
        ]
        '''
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # x.shape (784, 1)
        # y.shapep (10, 1)
        pdb.set_trace() 
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    
    # # z.txt
    # print 'z------------------------------------------------------------------'
    # print z
    # raw_input()

    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



    
# # mini_batch.txt
# print 'mini_batch<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
# print len(mini_batch)
# print 'mini_batch>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
# raw_input()