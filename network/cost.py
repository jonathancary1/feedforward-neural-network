import numpy as np

class CrossEntropy:
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = 1

    def backpropagate(self, design_matrix_hat, design_matrix):
        """Return the initial design matrix of gradient vectors."""
        return -design_matrix / design_matrix_hat

    def feedforward(self, design_matrix_hat, design_matrix):
        """Calculate the cross entropy between each pair of probability distributions."""
        return -np.sum(design_matrix * np.log(design_matrix_hat), axis=1)

class MeanSquaredError:
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = 1

    def backpropagate(self, design_matrix_hat, design_matrix):
        """Return the initial design matrix of gradient vectors."""
        return 2.0 * (design_matrix_hat - design_matrix)

    def feedforward(self, design_matrix_hat, design_matrix):
        """Calculate the mean squaared error between each pair of output vectors."""
        delta = design_matrix_hat - design_matrix
        return np.sum(delta * delta, axis=1)
