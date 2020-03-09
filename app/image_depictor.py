import numpy
from matplotlib import pyplot as plt


with open('../datasets/mnist_train_100.csv') as data_file:
    first_image_values = data_file.readline().split(',')
    print(f'Depicted number is {first_image_values[0]}')
    image_array = numpy.asfarray(first_image_values[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()
