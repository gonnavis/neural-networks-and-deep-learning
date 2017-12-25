
import mnist_loader
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()


# print len(training_data) # 50000
# print len(training_data[0]) # 2
# print len(training_data[1]) # 2
# print len(training_data[2]) # 2
# print len(training_data[0][0]) # 784
# # print len(training_data[0][1]) # error
# print     training_data[0][1] # 7
# print     training_data[1][1] # 2
# print len(training_data[0][0][0]) # 1
# print len(training_data[0][0][1]) # 1
# # print len(training_data[0][0][0][0]) # error

'''
training_data[ // 10000
  [
    [], // 784 image data
    [
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 1.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
    ],
  ],
  [
    [], // 784 image data
    [
      [ 1.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
      [ 0.],
    ],
  ],
]
'''


print len(validation_data) # 10000
print len(validation_data[0]) # 2
print len(validation_data[1]) # 2
print len(validation_data[2]) # 2
print len(validation_data[0][0]) # 784
# print len(validation_data[0][1]) # error
print     validation_data[0][1] # 3
print     validation_data[1][1] # 8
print len(validation_data[0][0][0]) # 1
print len(validation_data[0][0][1]) # 1
# print len(validation_data[0][0][0][0]) # error

'''
validation_data[ // 10000
  [
    [], // 784 image data
    3, // digit
  ],
  [
    [], // 784 image data
    8, // digit
  ],
]
'''


# print len(test_data) # 10000
# print len(test_data[0]) # 2
# print len(test_data[1]) # 2
# print len(test_data[2]) # 2
# print len(test_data[0][0]) # 784
# # print len(test_data[0][1]) # error
# print     test_data[0][1] # 7
# print     test_data[1][1] # 2
# print len(test_data[0][0][0]) # 1
# print len(test_data[0][0][1]) # 1
# # print len(test_data[0][0][0][0]) # error

'''
test_data[ // 10000
  [
    [], // 784 image data
    7, // digit
  ],
  [
    [], // 784 image data
    2, // digit
  ],
]
'''


raw_input()

import network
net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)