import numpy
from app.neural_network import NeuralNetwork


if __name__ == '__main__':
    input_nodes_number = 3
    hidden_nodes_number = 3
    output_nodes_number = 3

    learning_factor = 0.3
    # generating (hidden_nodes_number x input_nodes_number) matrix of
    # normally distributed random values (from -0.5 to 0.5)
    normal_distribution_center = 0.0
    w_input_hidden = numpy.random.normal(normal_distribution_center, pow(hidden_nodes_number, -0.5),
                                         (hidden_nodes_number, input_nodes_number))
    # generating (output_nodes_number x hidden_nodes_number) matrix of
    # normally distributed random values (from -0.5 to 0.5)
    w_hidden_output = numpy.random.normal(normal_distribution_center, pow(output_nodes_number, -0.5),
                                          (output_nodes_number, hidden_nodes_number))

    n = NeuralNetwork(input_nodes_number=input_nodes_number, hidden_nodes_number=hidden_nodes_number,
                      output_nodes_number=output_nodes_number, learning_factor=learning_factor,
                      w_input_hidden=w_input_hidden, w_hidden_output=w_hidden_output)

    print(n.w_input_hidden, n.w_hidden_output)
    print(n.query([1.0, 0.5, -1.5]))
