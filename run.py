import numpy
from app.neural_network import NeuralNetwork
from app.utils import get_normally_distributed_matrix

if __name__ == '__main__':
    input_nodes_number = 784  # number of pixels in 28x28 picture
    hidden_nodes_number = 100  # better to be in range >= 10*10 and < 784
    output_nodes_number = 10

    learning_factor = 0.3

    w_input_hidden = get_normally_distributed_matrix(hidden_nodes_number, input_nodes_number)
    w_hidden_output = get_normally_distributed_matrix(output_nodes_number, hidden_nodes_number)

    n = NeuralNetwork(input_nodes_number=input_nodes_number, hidden_nodes_number=hidden_nodes_number,
                      output_nodes_number=output_nodes_number, learning_factor=learning_factor,
                      w_input_hidden=w_input_hidden, w_hidden_output=w_hidden_output)

    # training the neural network
    with open('datasets/mnist_train.csv') as training_data_file:
        for record in training_data_file:
            separated_record_values = record.split(',')
            # scale the input data to the range 0.01 0.99
            inputs = (numpy.asfarray(separated_record_values[1:]) / 255.0 * 0.99) + 0.01
            # setting targeting values to 0.01 except of the desirable one which is 0.99
            targets = numpy.zeros(output_nodes_number) + 0.01
            targets[int(separated_record_values[0])] = 0.99
            n.train(inputs, targets)

    efficiency_scores = []
    # testing the efficiency of the neural network
    with open('datasets/mnist_test.csv') as test_data_file:
        for record in test_data_file:
            separated_record_values = record.split(',')
            correct_answer = int(separated_record_values[0])
            print(f'Corrected answer is {correct_answer}')

            inputs = (numpy.asfarray(separated_record_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = n.query(inputs)

            predicted_value = numpy.argmax(outputs)  # getting the index of the biggest element
            print(f'Predicted value is {predicted_value}')
            if correct_answer == predicted_value:
                efficiency_scores.append(1)
            else:
                efficiency_scores.append(0)

    efficiency_score = sum(efficiency_scores) / len(efficiency_scores)
    print(f'{efficiency_score:.2f}')
