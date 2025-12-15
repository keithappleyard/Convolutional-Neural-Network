import tarfile
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from utility import *

# Extract CIFAR-10 dataset
file = "cifar-10-python.tar.gz"

with tarfile.open(file, "r:gz") as tar:
    tar.extractall(path="./")

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

X = []
y = []

# Load each batch
path = "./cifar-10-batches-py/data_batch"
for i in range (1,6):
    # Restrict classes
    batch = unpickle(f'{path}_{i}')

    data = batch[b'data']
    labels = batch[b'labels']

    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    X.extend(data)
    y.extend(labels)

X = np.array(X)
y = np.array(y)

# Filter data to certain labels
selected_classes = [0,1,2,3,4]
mask = np.isin(y, selected_classes)

X_filtered = X[mask]
y_filtered = y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=38997053)

# Rearrange sliding image into columns of new matrix
def im2col_indices(x, field_height, field_width, stride=1, padding=0):
    batch, height, width, channels = x.shape
    h_padded, w_padded = height + 2*padding, width + 2*padding
    x_padded = np.pad(x, ((0,0), (padding,padding), (padding,padding), (0,0)), mode='constant')

    out_h = (h_padded - field_height) // stride + 1
    out_w = (w_padded - field_width) // stride + 1

    # Indices for im2col
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, channels)
    j0 = np.tile(np.arange(field_width), field_height)
    j0 = np.tile(j0, channels)
    k0 = np.repeat(np.arange(channels), field_height * field_width)

    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1,-1)
    j = j0.reshape(-1, 1) + j1.reshape(1,-1)
    k = k0.reshape(-1, 1)

    cols = x_padded[:, i, j, k] 

    cols = cols.transpose(1, 0, 2).reshape(field_height * field_width * channels, batch*out_h*out_w)

    return cols

def col2im_indices(cols, x_shape, field_height, field_width, stride=1, padding=0):  
    batch, height, width, channels = x_shape
    h_padded, w_padded = height + 2*padding, width + 2*padding
    x_padded = np.zeros((batch, h_padded, w_padded, channels), dtype=cols.dtype)

    out_h = (h_padded - field_height) // stride + 1
    out_w = (w_padded - field_width) // stride + 1

    # Indices for col2im
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, channels)
    j0 = np.tile(np.arange(field_width), field_height)
    j0 = np.tile(j0, channels)
    k0 = np.repeat(np.arange(channels), field_height * field_width)

    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = k0.reshape(-1, 1)

    i = np.tile(i, (1, batch))
    j = np.tile(j, (1, batch))

    b_idx = np.repeat(np.arange(batch), out_h * out_w).reshape(1, -1)

    cols_reshaped = cols.reshape(field_height*field_width*channels, -1)
    
    np.add.at(x_padded, (b_idx, i, j, k), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, padding:-padding, padding:-padding, :]

# Convolution layer using He initialization
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weight_decay=0.0):
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                 np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels, dtype=float)
        self.weight_decay = weight_decay

    def forward(self, x):
        self.x = x
        batch, height, width, channels = x.shape
        k = self.k

        self.cols = im2col_indices(x, k, k, stride=self.stride, padding=self.padding)

        # Flatten
        W_col = self.W.reshape(self.out_channels, -1)
        out = np.dot(W_col, self.cols) + self.b.reshape(-1, 1)

        out_h = (height + 2*self.padding - k) // self.stride + 1
        out_w = (width + 2*self.padding - k) // self.stride + 1
        out = out.reshape(self.out_channels, batch, out_h, out_w)

        return out.transpose(1, 2, 3, 0)
    
    def backward(self, grad, lr):
        grad_flat = grad.transpose(3,0,1,2).reshape(self.out_channels, -1)

        db = np.sum(grad_flat, axis=1)

        dW = np.dot(grad_flat, self.cols.T)
        dW = dW.reshape(self.W.shape)
        dW += self.weight_decay * self.W

        # Flatten
        W_col = self.W.reshape(self.out_channels, -1)
        dcols = np.dot(W_col.T, grad_flat)

        dx = col2im_indices(dcols, self.x.shape, self.k, self.k, stride=self.stride, padding=self.padding)
        
        self.W -= lr * dW
        self.b -= lr * db

        return dx

# Batch Normalisation implementation
class BatchNorm:
    def __init__(self, channels, momentum=0.9):
        self.gamma = np.ones((1, 1, 1, channels))
        self.beta = np.zeros((1, 1, 1, channels))
        self.momentum = momentum
        self.running_mean = np.zeros((1, 1, 1, channels))
        self.running_var = np.ones((1, 1, 1, channels))

    def forward(self, x, training=True):
        if training:
            self.mean = np.mean(x, axis=(0,1,2), keepdims=True)
            self.var = np.var(x, axis=(0,1,2), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var

        self.x = x
        self.norm = (self.x - self.mean) / np.sqrt(self.var + 1e-8)
        
        return self.gamma * self.norm + self.beta

    def backward(self, grad, lr):
        batch, height, width, channels = grad.shape

        dgamma = np.sum(grad * self.norm, axis=(0,1,2), keepdims=True)
        dbeta = np.sum(grad, axis=(0,1,2), keepdims=True)

        dx_norm = grad * self.gamma
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 * (self.var + 1e-8)**(-1.5), axis=(0,1,2), keepdims=True)
        dmean = np.sum(-dx_norm / np.sqrt(self.var + 1e-8) , axis=(0,1,2), keepdims=True) + dvar * np.sum(-2*(self.x - self.mean), axis=(0,1,2), keepdims=True) / (batch*height*width)
        dx = dx_norm / np.sqrt(self.var + 1e-8) + dvar * 2*(self.x - self.mean) / (batch*height*width) + dmean / (batch*height*width)

        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta

        return dx

# Activation layer (ReLU)
class Activation:
    def forward(self, X):
        self.X = X
        return ReLU(X)
    
    def backward(self, d_out):
        return d_out * ReLU_derivative(self.X)

# Pooling layer
class MaxPool: # 2x2, stride=2
    def __init__(self):
        self.pool_height = 2
        self.pool_width = 2
        self.stride = 2

    def forward(self,x):
        self.x = x
        batch, height, width, channels = x.shape
        
        cols = im2col_indices(
            x,
            self.pool_height,
            self.pool_width,
            self.stride,
            padding=0
        )

        windows = cols.shape[1]

        cols_reshaped = cols.reshape(channels, 4, windows)

        out = np.max(cols_reshaped, axis=1)

        self.max_mask = (cols_reshaped == out[:, None, :])

        out_h, out_w = height // 2, width // 2
        
        out = out.reshape(channels, batch, out_h, out_w)
        out = out.transpose(1,2,3,0)

        return out

    def backward(self, grad):
        batch, height, width, channels = self.x.shape
        out_h, out_w = height // 2, width // 2

        grad_flat = grad.transpose(3,0,1,2).reshape(channels, -1)

        dcols = (self.max_mask * grad_flat[:, None, :])
        dcols = dcols.reshape(4*channels, -1)

        dx = col2im_indices(
            dcols,
            self.x.shape,
            self.pool_height,
            self.pool_width,
            self.stride,
            padding=0
        )

        return dx

# Fully Connected layer
class FullyConnected:
    def __init__(self, in_size, out_size, weight_decay=0.0):
        self.W = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
        self.b = np.zeros((1, out_size))
        self.weight_decay = weight_decay
    
    def forward(self, x):
        self.original_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b
    
    def backward(self, grad, lr):
        dW = np.dot(self.x.T, grad) + self.weight_decay * self.W
        db = np.sum(grad, axis=0, keepdims=True)
        dx = np.dot(grad, self.W.T)

        self.W -= lr * dW
        self.b -= lr * db

        return dx.reshape(self.original_shape)

# Dropout implementation
class Dropout:
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = True
    
    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.rate)
            return x * self.mask / (1 - self.rate)
        return x
    
    def backward(self, grad):
        return grad * self.mask

# Network initialization variables
n_outputs = 5 # Number of outputs
lr = 0.01 # Learning rate
max_iterations = 8 # Maximum number of iterations

# Batch variables
batch_size = 32
n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

# List of loss history and accuracy
train_loss_array = []
val_loss_array = []

train_accuracy_array = []
val_accuracy_array = []


# Flatten and encode data
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
y_train_encoded = one_hot(y_train, n_outputs)
y_test_encoded = one_hot(y_test, n_outputs)

# Network Architecture
layers = [
    # Block 1
    Conv2D(3, 16, kernel_size=3, stride=1, padding=1, weight_decay=1e-4),
    BatchNorm(16),
    Activation(),
    MaxPool(),

    # Block 2
    Conv2D(16, 32, kernel_size=3, stride=1, padding=1, weight_decay=1e-4),
    BatchNorm(32),
    Activation(),
    MaxPool(),

    # Block 3 (Fully Connected)
    FullyConnected(32 * 8 * 8, 128, weight_decay=1e-4),
    Activation(),
    Dropout(),

    # Output layer
    FullyConnected(128, 5) # Produces logits
]

# Forward without trainnig
def predict(X):
    for layer in layers:
        if isinstance(layer, (BatchNorm, Dropout)):
            X = layer.forward(X, training=False)
        else:
            X = layer.forward(X)
    return softmax(X)

# Training Loop
for epoch in range(max_iterations):
    idx = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[idx]
    y_train_shuffled = y_train_encoded[idx]

    epoch_loss = 0
    epoch_accuracy = 0
    epoch_correct = 0
    epoch_total = 0

    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size

        Xb = X_train_shuffled[start:end]
        yb = y_train_shuffled[start:end]

        # Forward pass
        value = Xb
        for layer in layers:
            if isinstance(layer, (BatchNorm, Dropout)):
                value = layer.forward(value, training=True)
            else:
                value = layer.forward(value)
        
        pred = softmax(value)

        # Compute loss and accuracy metrics per batch
        epoch_loss += cross_entropy_loss(pred, yb)

        epoch_correct += np.sum(np.argmax(pred, axis=1) == np.argmax(yb, axis=1))
        epoch_total += Xb.shape[0]

        # Backpropagation
        grad = (pred - yb) / batch_size

        for layer in reversed(layers):
            if isinstance(layer, (Dropout, Activation, MaxPool)):
                grad = layer.backward(grad)
            else:
                grad = layer.backward(grad, lr)

    # Compute training accuracy and loss per epoch
    loss = epoch_loss / n_batches
    train_loss_array.append(loss)

    epoch_acc = epoch_correct / epoch_total
    train_accuracy_array.append(epoch_acc*100)

    # Compute validation accuracy and loss per epoch
    y_pred = predict(X_test)
    val_acc = accuracy(np.argmax(y_pred, axis=1), y_test)
    val_accuracy_array.append(val_acc*100)

    val_loss = cross_entropy_loss(y_pred, y_test_encoded)
    val_loss_array.append(val_loss)

    print(f"Epoch {1+epoch:4d} | Loss: {loss:.6f} | Training Accuracy: {epoch_acc*100:.2f}% | Test accuracy: {val_acc*100:.2f}%")
