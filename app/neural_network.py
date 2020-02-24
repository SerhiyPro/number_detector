class NeuralNetwork:
    # TODO: use dataclasses
    def __init__(self, *args, **kwargs):
        self.input_nodes_number = kwargs.get('input_nodes_number', 0)
        self.hidden_nodes_number = kwargs.get('hidden_nodes_number', 0)
        self.output_nodes_number = kwargs.get('output_nodes_number', 0)
        self.learning_factor = kwargs.get('learning_factor', 0)

    def train(self):
        pass

    def query(self):
        pass
