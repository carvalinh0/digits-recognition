import numpy as np
from layer import Layer

class Train:
    def __init__(self, layers: list[Layer], train_set: tuple, loss_function: callable, loss_function_derivative: callable):
        self.layers = layers
        self.train_set = train_set
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def run(self) -> int:
        train_count = 0
        for image_index, image_pixels in enumerate(self.train_set[0]): 
            
            # 1. Define the output as the input pixels of the image for the first layer
            output = image_pixels
            
            # 2. Forward pass through the network
            for layer in self.layers:
                output = layer.forward(output)
            
            # 3. Create the target vector for the correct label
            correct_label = self.train_set[1][image_index]
            target_vector = np.zeros((1, 10))
            target_vector[0, correct_label] = 1.0

            # 4. Calculate the loss
            loss = self.loss_function(output, target_vector)

            # 5. Backpropagation and weight updates
            gradient = self.loss_function_derivative(output, target_vector)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient)

            train_count += 1
        return train_count