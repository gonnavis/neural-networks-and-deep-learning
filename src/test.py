
import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

import network
net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# check one digit
# import numpy as np
# np.argmax(net.feedforward(test_data[0][0]))
# execfile('C:\\test.py')
# %windir%\System32\cmd.exe "/K" d:\ProgramData\Anaconda2\Scripts\activate.bat d:\ProgramData\Anaconda2
# arr.shape