from itertools import chain, tee
from network.cost import MeanSquaredError

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class FeedforwardNeuralNetwork:
    def __init__(self, *layers, cost=None):
        if cost is None:
            cost = MeanSquaredError(layers[-1].output_dimension)
        for a, b in pairwise(chain(layers, [cost])):
            if a.output_dimension != b.input_dimension:
                raise ValueError("input_dimension must equal output_dimension of the preceding layer")
        if cost.output_dimension != 1:
            raise ValueError("output_dimension of cost must equal 1")
        self.cost = cost
        self.layers = layers

    def activations(self, design_matrix):
        """Return a tensor of activations for each layer of the network."""
        activations = [design_matrix]
        for layer in self.layers:
            design_matrix = layer.feedforward(design_matrix)
            activations.append(design_matrix)
        return activations

    def backpropagate(self, activations, design_matrix, learning_rate=0.1):
        """Perform gradient descent using backpropagation."""
        gradients = self.cost.backpropagate(activations[-1], design_matrix)
        learning_rate /= len(gradients)
        for layer, activations in zip(reversed(self.layers), reversed(activations[:-1])):
            gradients = layer.backpropagate(activations, gradients, learning_rate)

    def error(self, design_matrix_hat, design_matrix):
        """Calculate the cost for each row of the design matrix of output vectors."""
        return self.cost.feedforward(design_matrix_hat, design_matrix)

    def feedforward(self, design_matrix):
        """Calculate the network output for each row of the design matrix of inputs vectors."""
        for layer in self.layers:
            design_matrix = layer.feedforward(design_matrix)
        return design_matrix
