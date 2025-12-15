import numpy as np

# Implement cross entropy loss function
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

# Implement accuracy function
def accuracy(y_pred, y_true):
    return np.mean(y_true == y_pred)

# Implement ReLU activation function
def ReLU(x):
    return np.maximum(0, x)

# Implement ReLU derivative function
def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

# Implement softmax activation function
def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Encode using one hot encoding
def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out