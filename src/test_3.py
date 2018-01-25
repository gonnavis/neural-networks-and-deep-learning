
import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()


''' training_data len(5000)
[
  ( array(784, 1), array(10, 1) ),
]
'''

''' validation_data len(10000)
[
  ( array(784, 1), 0 ),
  ( array(784, 1), 1 ),
  ( array(784, 1), 9 ),
]
'''

''' test_data len(10000)
[
  ( array(784, 1), 0 ),
  ( array(784, 1), 1 ),
  ( array(784, 1), 9 ),
]
'''


import network
net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# check one digit
# import numpy as np
# np.argmax(net.feedforward(test_data[0][0]))
# np.argmax(net.feedforward( ))
# execfile('C:\\test.py')