import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.activations import softmax
from utils.loss import cross_entropy_loss


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


class CNN:
    def __init__(self, input_shape=(28, 28), kernel_size=3, output_size=10):
        self.kernel_size = kernel_size
        self.kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32) * 0.1
        self.bias_conv = 0.0

        conv_out_dim = input_shape[0] - kernel_size + 1  # no padding, stride=1
        self.fc_input_size = conv_out_dim * conv_out_dim

        self.W_fc = np.random.randn(self.fc_input_size, output_size).astype(np.float32) * 0.1
        self.b_fc = np.zeros((1, output_size), dtype=np.float32)

    # ------------------------------ forward ------------------------------
    def convolve2d(self, image, kernel):
        kh, kw = kernel.shape
        ih, iw = image.shape
        out = np.zeros((ih - kh + 1, iw - kw + 1), dtype=np.float32)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
        return out

    def forward(self, X):
        self.batch_size = X.shape[0]
        self.conv_out = np.array([self.convolve2d(img, self.kernel) + self.bias_conv for img in X], dtype=np.float32)
        self.relu_out = relu(self.conv_out)
        self.flat = self.relu_out.reshape(self.batch_size, -1)
        self.logits = np.dot(self.flat, self.W_fc) + self.b_fc
        probs = softmax(self.logits)
        self.probs = np.clip(probs, 1e-8, 1 - 1e-8)  # numerical stability
        return self.probs

    # ------------------------------ backward -----------------------------
    def backward(self, X, y, output):
        m = X.shape[0]
        d_logits = output - y                       # [m, 10]
        dW_fc = np.dot(self.flat.T, d_logits) / m   # [fc_in,10]
        db_fc = np.sum(d_logits, axis=0, keepdims=True) / m

        d_flat = np.dot(d_logits, self.W_fc.T)                      # [m, fc_in]
        d_relu = d_flat.reshape(self.relu_out.shape) * relu_derivative(self.conv_out)

        # gradient w.r.t kernel
        d_kernel = np.zeros_like(self.kernel)
        for i in range(m):
            img = X[i]
            err = d_relu[i]
            for h in range(self.kernel.shape[0]):
                for w in range(self.kernel.shape[1]):
                    patch = img[h:h + err.shape[0], w:w + err.shape[1]]
                    d_kernel[h, w] += np.sum(patch * err)
        d_kernel /= m
        d_bias_conv = np.sum(d_relu) / m

        # parameter update
        self.W_fc -= self.lr * dW_fc
        self.b_fc -= self.lr * db_fc
        self.kernel -= self.lr * d_kernel
        self.bias_conv -= self.lr * d_bias_conv

    # ------------------------------ training helpers --------------------
    def train(self, loader_fn, epochs=5, lr=0.001):
        """`loader_fn` must return a *fresh* generator each epoch."""
        self.lr = lr
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, y_batch in loader_fn():
                out = self.forward(X_batch)
                loss = cross_entropy_loss(out, y_batch)
                total_loss += loss
                self.backward(X_batch, y_batch, out)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, loader):
        correct = total = 0
        for X_batch, y_batch in loader:
            preds = self.predict(X_batch)
            labels = np.argmax(y_batch, axis=1)
            correct += (preds == labels).sum()
            total += len(y_batch)
        print(f"Test Accuracy: {correct / total * 100:.2f}%")


# ------------------------------ utility ------------------------------

def one_hot(labels, num_classes=10):
    return np.eye(num_classes, dtype=np.float32)[labels]


# ------------------------------ main ------------------------------

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_ds = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader_pt = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader_pt = DataLoader(test_ds, batch_size=128, shuffle=False)

    # wrapper that yields NumPy batches (fresh generator each call)
    def to_numpy_loader(pt_loader):
        for imgs, labels in pt_loader:
            yield imgs.squeeze().numpy(), one_hot(labels.numpy())

    model = CNN()
    model.train(lambda: to_numpy_loader(train_loader_pt), epochs=5, lr=0.001)
    model.evaluate(to_numpy_loader(test_loader_pt))


if __name__ == '__main__':
    main()
