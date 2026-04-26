# Handwritten Digit Recognition using Feedforward Neural Networks

## Abstract

This project implements a fully-connected feedforward neural network trained on the MNIST dataset for handwritten digit classification. The network employs backpropagation for weight optimization and achieves 94.08% accuracy on the validation set and 93.48% on the test set. The implementation is built from scratch in Python using NumPy, providing transparency into the learning mechanisms of deep neural networks.

## 1. Introduction

Handwritten digit recognition is a foundational problem in machine learning and computer vision. The MNIST (Modified National Institute of Standards and Technology) dataset has served as a benchmark for evaluating neural network architectures and training algorithms since its introduction. This project demonstrates a custom implementation of a feedforward neural network trained with gradient descent backpropagation, showing how fundamental deep learning principles can be effectively applied to achieve competitive performance.

## 2. Methodology

### 2.1 Network Architecture

The neural network consists of three fully-connected layers that follows the [3blue1brown](https://www.youtube.com/watch?v=IHZwWFHWa-w) suggestions for this project:

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer 1**: 16 neurons with sigmoid activation
- **Hidden Layer 2**: 16 neurons with sigmoid activation  
- **Output Layer**: 10 neurons with sigmoid activation (one per digit class 0-9)

The architecture can be expressed as: $784 \rightarrow 16 \rightarrow 16 \rightarrow 10$

### 2.2 Activation Function

The sigmoid activation function is used throughout the network:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

The sigmoid function introduces non-linearity into the network, enabling it to learn complex patterns. Its derivative is utilized during backpropagation:

$$\frac{d\sigma}{dx} = \sigma(x)(1 - \sigma(x))$$

### 2.3 Forward Propagation

For each layer $l$, the forward pass is computed as:

$$z^{[l]} = a^{[l-1]} \cdot W^{[l]} + b^{[l]}$$

$$a^{[l]} = \sigma(z^{[l]})$$

where:
- $a^{[l]}$ represents the activations (outputs) of layer $l$
- $W^{[l]}$ are the weight matrices
- $b^{[l]}$ are the bias vectors
- $z^{[l]}$ represents the pre-activation values

### 2.4 Loss Function

Mean Squared Error (MSE) is employed as the loss function:

$$L = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2$$

where $y_i$ is the target vector and $\hat{y}_i$ is the network's prediction for sample $i$.

The loss gradient is:

$$\frac{\partial L}{\partial \hat{y}} = \frac{2}{m}(\hat{y} - y)$$

### 2.5 Backpropagation Algorithm

The backpropagation algorithm computes gradients for all weights and biases by chain rule application:

1. **Output Layer Gradient**:
$$\delta^{[L]} = \frac{\partial L}{\partial a^{[L]}} \odot \frac{\partial a^{[L]}}{\partial z^{[L]}}$$

2. **Hidden Layer Gradients**:
$$\delta^{[l]} = (\delta^{[l+1]} \cdot (W^{[l+1]})^T) \odot \sigma'(z^{[l]})$$

3. **Weight and Bias Gradients**:
$$\frac{\partial L}{\partial W^{[l]}} = (a^{[l-1]})^T \cdot \delta^{[l]}$$

$$\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}$$

4. **Parameter Updates** (Gradient Descent):
$$W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial L}{\partial W^{[l]}}$$

$$b^{[l]} \leftarrow b^{[l]} - \eta \frac{\partial L}{\partial b^{[l]}}$$

where $\eta$ is the learning rate (set to 1.0 in this implementation).

### 2.6 Training Strategy

The training employs the following strategies:

**Checkpoint Mechanism**: The best weights are saved whenever validation accuracy improves, allowing recovery of optimal parameters.

**Early Stopping with Threshold**: Training continues until:
- At least 1,000,000 training samples have been processed, AND
- Validation accuracy has not improved for a full epoch through the training set

This hybrid approach prevents premature stopping while avoiding overfitting.

### 2.7 Dataset

The MNIST dataset contains:
- **Training Set**: 50,000 images (28×28 pixels)
- **Validation Set**: 10,000 images
- **Test Set**: 10,000 images

Images are normalized to [0, 1] range and flattened to 784-dimensional vectors.

## 3. Implementation Details

### 3.1 Code Structure

```
layer.py          - Layer class implementing forward and backward passes
train.py          - Training loop implementation
validade.py       - Validation/evaluation module
main.py           - Main script orchestrating the training process
```

### 3.2 Key Classes

**Layer Class** (`layer.py`):
- Manages weights and biases with Xavier initialization (scaled random normal)
- Implements forward pass with sigmoid activation
- Implements backward pass for gradient computation
- Updates parameters using gradient descent

**Train Class** (`train.py`):
- Iterates through training samples
- Performs forward propagation
- Constructs one-hot encoded target vectors
- Executes backpropagation and parameter updates
- Returns number of samples processed

**Validade Class** (`validade.py`):
- Evaluates network on validation/test sets
- Computes accuracy as the fraction of correct predictions
- Uses argmax for classification decision

## 4. Results

### 4.1 Training Performance

| Metric | Value |
|--------|-------|
| Total Training Samples Processed | 1,050,000 |
| Best Validation Accuracy | 94.08% |
| Final Validation Accuracy | 93.98% |
| Test Set Accuracy | 93.48% |

### 4.2 Learning Curve

The validation accuracy improved progressively during training:
- After 50,000 samples: 40.41%
- After 100,000 samples: 87.07%
- After 150,000 samples: 91.54%
- After 250,000 samples: 92.72%
- After 750,000 samples: 94.08% (best)
- After 1,050,000 samples: 93.98% (final)

## 5. Discussion

### 5.1 Performance Analysis

The 93.48% test accuracy represents solid performance on this benchmark task. The modest gap between validation (93.98%) and test accuracy (93.48%) suggests minimal overfitting, indicating the model generalizes reasonably well to unseen data.

### 5.2 Training Dynamics

The learning curve shows rapid initial improvement followed by slower convergence, which is typical of gradient-based optimization. The plateau in accuracy improvements after approximately 700,000 samples suggests the network has approached its capacity with the current configuration.

## 6. Usage

### Prerequisites
- Python 3.7+
- NumPy
- Matplotlib
- MNIST dataset (`data/mnist.pkl.gz`)

You can also just run the following command to install the packages (virtual environment is recommended)

```bash
pip install -r requirements.txt
```

### Running the Training

```bash
python main.py
```

The script will:
1. Load the MNIST dataset
2. Initialize the neural network
3. Train the network with checkpointing and early stopping
4. Display training progress and metrics
5. Generate a validation accuracy plot
6. Report final test set performance

### Output

The training process prints:
- Accuracy improvements with checkpoint updates
- Final training statistics
- Best, final, and test accuracies
- A matplotlib graph of validation accuracy over time

## 7. Technical Notes

### 7.1 Numerical Stability

The sigmoid activation with MSE loss was chosen for simplicity. For production systems, softmax with cross-entropy loss would be more numerically stable for multi-class classification.

### 7.2 Performance Considerations

- Training processes all 50,000 samples per epoch
- Validation is performed after each complete epoch
- The implementation prioritizes clarity over computational efficiency
- GPU acceleration is not utilized (CPU-only implementation)

## 8. References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.

2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

4. 3Blue1Brown. (2017). But what is a neural network? [Video](https://www.youtube.com/watch?v=IHZwWFHWa-w). YouTube. https://www.youtube.com/watch?v=IHZwWFHWa-w

## 9. Conclusion

This project successfully demonstrates the implementation of a feedforward neural network for handwritten digit recognition from first principles. With 93.48% test accuracy, the network achieves competitive performance while maintaining code clarity and educational value. The implementation effectively illustrates fundamental concepts in deep learning including forward propagation, backpropagation, loss calculation, and gradient-based optimization.

---

**Author**: Felipe Carvalho 
**Date**: April 26, 2026  
**Implementation Language**: Python 3  
**Framework**: NumPy (no external ML frameworks)
