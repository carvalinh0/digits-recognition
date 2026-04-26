import numpy as np

# Define a simple feedforward neural network layer
class Layer: 
    def __init__(self, input_size, neurons_size, activation_function: callable, learning_rate=1): 
        self.input_size          = input_size
        self.neurons_size        = neurons_size
        self.weights             = np.random.randn(self.input_size, self.neurons_size) * 0.01
        self.biases              = np.zeros((1, self.neurons_size))
        self.activation_function = activation_function
        self.learning_rate       = learning_rate
        self.inputs              = None
        self.outputs             = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation_function(np.dot(inputs, self.weights) + self.biases)
        return self.outputs
    
    def backward(self, output_gradients):
        
        # Calculate the gradient of the activation function
        activation_derivative = self.outputs * (1.0 - self.outputs)
        delta = output_gradients * activation_derivative

        # Calculate gradients
        weights_gradient = np.outer(self.inputs, delta)
        input_gradient = np.dot(delta, self.weights.T)
        
        # Update weights and biases
        self.weights -= self.learning_rate * weights_gradient
        self.biases -= self.learning_rate * delta

        return input_gradient