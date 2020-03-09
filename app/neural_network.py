import numpy
import scipy.special


class NeuralNetwork:
    # TODO: use dataclasses
    def __init__(self, *args, **kwargs):
        self.input_nodes_number = kwargs.get('input_nodes_number', 0)
        self.hidden_nodes_number = kwargs.get('hidden_nodes_number', 0)
        self.output_nodes_number = kwargs.get('output_nodes_number', 0)
        self.learning_factor = kwargs.get('learning_factor', 0)
        self.w_input_hidden = kwargs.get('w_input_hidden')
        self.w_hidden_output = kwargs.get('w_hidden_output')

    def train(self, input_layer: list, target_layer: list):
        # convert input lists into two-dimensional matrix
        inputs = numpy.array(input_layer, ndmin=2).T
        targets = numpy.array(target_layer, ndmin=2).T

        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_outputs = self.query(inputs)
        output_errors = targets - final_outputs
        #  hidden_layer_errors = transposed matrix of weighting factors * errors of output layer
        hidden_errors = numpy.dot(self.w_hidden_output.T, output_errors)

        # refresh weighting factors between hidden and output layers
        self.w_hidden_output += self.learning_factor * numpy.dot((output_errors * final_outputs * (1.0-final_outputs)),
                                                                 numpy.transpose(hidden_outputs))
        # refresh weighting factors between input and hidden layers
        self.w_input_hidden += self.learning_factor * numpy.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)),
                                                                numpy.transpose(inputs))

    def query(self, input_layer: list) -> numpy.ndarray:
        hidden_inputs = numpy.dot(self.w_input_hidden, input_layer)  # X_hidden = W_inputHidden * I
        hidden_outputs = self.activation_function(hidden_inputs)  # O_hidden = sigmoid(X_hidden)

        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)  # X_final = W_hiddenOutput * O_hidden
        return self.activation_function(final_inputs)  # O_final = sigmoid(X_final)

    @staticmethod
    def activation_function(x) -> numpy.ndarray:
        """
        Sigmoid function (1/1+e^(-x))
        """
        return scipy.special.expit(x)
