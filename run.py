from app.neural_network import NeuralNetwork


if __name__ == '__main__':
    input_nodes_number = 3
    hidden_nodes_number = 3
    output_nodes_number = 3

    learning_factor = 0.3

    n = NeuralNetwork(input_nodes_number=input_nodes_number, hidden_nodes_number=hidden_nodes_number,
                      output_nodes_number=output_nodes_number, learning_factor=learning_factor)
