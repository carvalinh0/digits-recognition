import numpy as np

class Validade:
    def __init__(self, layers: list, valid_set: tuple):
        self.valid_set = valid_set
        self.layers = layers

    def run(self) -> float:
        actual_accuracy = 0.0
        for image_index, image_pixels in enumerate(self.valid_set[0]): 
            
            # 1. Define the output as the input pixels of the image for the first layer
            image_pixels = image_pixels.reshape(1, -1)
            output = image_pixels
            
            # 2. Forward pass through the network
            for layer in self.layers:
                output = layer.forward(output)

            # Calculate the accuracy
            accuracy = np.argmax(output) == self.valid_set[1][image_index]
            actual_accuracy += accuracy

        return actual_accuracy / len(self.valid_set[0])