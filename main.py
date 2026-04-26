import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from layer import Layer
from train import Train
from validade import Validade

# Load the MNIST dataset
with gzip.open('data/mnist.pkl.gz', 'rb') as f: 
    # image set: (50000, 784) as set[0]
    # label set: (50000,) as set[1]

    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# Define the sigmoid activation function
def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def calculate_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def calculate_loss_derivative(predictions, targets):
    return 2 * (predictions - targets) / targets.size

def save_checkpoint(layers):
    return [(layer.weights.copy(), layer.biases.copy()) for layer in layers]

def load_checkpoint(layers, checkpoint):
    for layer, (saved_weights, saved_biases) in zip(layers, checkpoint):
        layer.weights = saved_weights.copy()
        layer.biases = saved_biases.copy()

# Create the layers of the network
hidden_layer1 = Layer(784, 16, sigmoid, 1)                         # First hidden layer with 16 neurons
hidden_layer2 = Layer(hidden_layer1.neurons_size, 16, sigmoid, 1)  # Second hidden layer with 16 neurons
output_layer  = Layer(hidden_layer2.neurons_size, 10, sigmoid, 1)  # Output layer with 10 neurons (for digits 0-9)
layers        = [hidden_layer1, hidden_layer2, output_layer]

if __name__ == "__main__":
    
    actual_accuracy = 0
    accuracy        = [0.0]
    train_count     = 0

    best_checkpoint = save_checkpoint(layers) # Save the initial state of the network as a checkpoint

    trainer   = Train(layers, train_set, calculate_loss, calculate_loss_derivative)
    validator = Validade(layers, valid_set)

    train_at_least_x_times = 1_000_000

    while True:
        # 1. Train the network on the training set
        train_count += trainer.run()

        # 2. Validate the network on the validation set and calculate the accuracy
        actual_accuracy = validator.run()

        # Check if the accuracy has increased and stop training if it has decreased (not the best way to do it but it's a simple approach for this example)
        if actual_accuracy > accuracy[-1]:
            accuracy.append(actual_accuracy)
            best_checkpoint = save_checkpoint(layers) # Update the checkpoint with the new weights
            print(f"Accuracy improved to {actual_accuracy * 100:.4f}%. Checkpoint updated.")
            print("Actual training count:", train_count)

        # Continue training if the accuracy has not improved but we haven't trained at least X times yet
        elif train_count < train_at_least_x_times:
            accuracy.append(actual_accuracy)
            print(f"Accuracy: {actual_accuracy * 100:.4f}%. Pushing through...")
        
        # Stopping training if accuracy has decreased and don't update the checkpoint
        else:
            load_checkpoint(layers, best_checkpoint)  # Load the best checkpoint to restore the previous weights
            break

    print("-"*50)
    print("Training completed.")
    print("Training count:", train_count)
    print(f"Last accuracy: {actual_accuracy * 100:.4f}%")
    print(f"Best accuracy: {max(accuracy) * 100:.4f}%")
    test_accuracy = Validade(layers, test_set).run()
    print(f"Test accuracy: {test_accuracy * 100:.4f}%")

    plt.plot(accuracy)
    plt.title("Validation Accuracy Over Time")
    plt.xlabel("Training Iterations")
    plt.ylabel("Validation Accuracy")
    plt.grid()
    plt.show()