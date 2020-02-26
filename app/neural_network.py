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

    def train(self):
        pass

    def query(self, input_layer: list):
        hidden_inputs = numpy.dot(self.w_input_hidden, input_layer)  # X_hidden = W_inputHidden * I
        hidden_outputs = self.activation_function(hidden_inputs)  # O_hidden = sigmoid(X_hidden)

        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)  # X_final = W_hiddenOutput * O_hidden
        return self.activation_function(final_inputs)  # O_final = sigmoid(X_final)

    @staticmethod
    def activation_function(x) -> float:
        """
        Sigmoid function (1/1+e^(-x))
        """
        return scipy.special.expit(x)
