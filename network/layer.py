import numpy as np

class AffineTransformationLayer:
    def __init__(self, input_dimension, output_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        # random weights and biases are assigned in order to break symmetry
        self.weights = np.random.normal(0.0, np.sqrt(1.0 / input_dimension), (output_dimension, input_dimension))
        self.biases = np.zeros(output_dimension)

    def backpropagate(self, activations, gradients, epsilon):
        """Perform gradient descent and return the new design matrix of gradients for the previous layer."""
        backpropagation = gradients @ self.weights
        self.biases -= np.sum(epsilon * gradients, axis=0)
        self.weights -= np.sum(epsilon * gradients[:, :, np.newaxis] * activations[:, np.newaxis, :], axis=0)
        return backpropagation

    def feedforward(self, activations):
        """Apply an affine transformation to each row of the design matrix of activations."""
        return activations @ np.transpose(self.weights) + self.biases

class ReluLayer:
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = input_dimension

    def backpropagate(self, activations, gradients, epsilon):
        """Return the new design matrix of gradients for the previous layer."""
        return (activations > 0.0) * gradients

    def feedforward(self, activations):
        """Apply a rectifier to each row of the design matrix of activations."""
        return (activations > 0.0) * activations

class SoftmaxLayer:
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.output_dimension = input_dimension

    def backpropagate(self, activations, gradients, epsilon):
        """Return the new design matrix of gradients for the previous layer."""
        softmax = self.feedforward(activations)
        diagonals = np.array([np.diag(i) for i in softmax])
        jacobians = diagonals - softmax[:, :, np.newaxis] * softmax[:, np.newaxis, :]
        backpropagation = np.matmul(jacobians, gradients[:, :, np.newaxis])
        return backpropagation.reshape((len(gradients), self.input_dimension))

    def feedforward(self, activations):
        """Apply softmax to each row of the design matrix of activations."""
        unnormalized = np.exp(activations - np.max(activations, axis=1)[:, np.newaxis])
        return unnormalized / np.sum(unnormalized, axis=1)[:, np.newaxis]
