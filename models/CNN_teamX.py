import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

# Logging setup
logging.basicConfig(filename="cnn_training_log.txt", level=logging.INFO, filemode="w")

def relu(x):
    logging.info(f"Calling ReLU activation: input shape {x.shape}")
    out = np.maximum(0, x)
    logging.info(f"ReLU activation completed: output shape {out.shape}")
    return out

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    m = predictions.shape[0]
    clipped_preds = np.clip(predictions, 1e-12, 1 - 1e-12)
    loss = -np.sum(targets * np.log(clipped_preds)) / m
    logging.info(f"Cross entropy loss computed: {loss:.6f}")
    return loss

def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def extract_patches(X, kernel_size):
    batch, h, w = X.shape
    out_h = h - kernel_size + 1
    out_w = w - kernel_size + 1
    shape = (batch, out_h, out_w, kernel_size, kernel_size)
    strides = (
        X.strides[0], X.strides[1], X.strides[2], X.strides[1], X.strides[2]
    )
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

class CNN:
    def __init__(self, input_shape=(28, 28), num_kernels=8, kernel_size=3, output_size=10):
        logging.info("Initializing CNN model")
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.kernels = np.random.randn(num_kernels, kernel_size, kernel_size) * 0.01
        self.bias_conv = np.zeros((num_kernels, 1))

        conv_dim = input_shape[0] - kernel_size + 1
        self.conv_out_dim = conv_dim
        self.fc_input_size = num_kernels * conv_dim * conv_dim
        self.W_fc = np.random.randn(self.fc_input_size, output_size) * 0.01
        self.b_fc = np.zeros((1, output_size))

    def forward(self, X):
        logging.info(f"CNN forward pass: input batch shape {X.shape}")
        self.X = X
        patches = extract_patches(X, self.kernel_size)
        self.patches = patches

        conv_out = np.zeros((X.shape[0], self.num_kernels, self.conv_out_dim, self.conv_out_dim))
        for k in range(self.num_kernels):
            kernel = self.kernels[k]
            conv_out[:, k] = np.tensordot(patches, kernel, axes=((3, 4), (0, 1))) + self.bias_conv[k]
        self.conv_out = conv_out
        self.relu_out = relu(conv_out)
        self.flat = self.relu_out.reshape(X.shape[0], -1)
        self.logits = np.dot(self.flat, self.W_fc) + self.b_fc
        self.probs = softmax(self.logits)
        logging.info(f"CNN forward pass completed: output shape {self.probs.shape}")
        return self.probs

    def backward(self, y_true):
        m = y_true.shape[0]
        d_logits = self.probs - y_true
        dW_fc = np.dot(self.flat.T, d_logits) / m
        db_fc = np.sum(d_logits, axis=0, keepdims=True) / m

        d_flat = np.dot(d_logits, self.W_fc.T).reshape(self.relu_out.shape)
        d_relu = d_flat * relu_derivative(self.conv_out)

        d_kernels = np.zeros_like(self.kernels)
        d_bias_conv = np.sum(d_relu, axis=(0, 2, 3), keepdims=True).reshape(-1, 1)

        for k in range(self.num_kernels):
            grad = d_relu[:, k]
            grad_expanded = grad[:, :, :, np.newaxis, np.newaxis]
            d_kernels[k] = np.sum(self.patches * grad_expanded, axis=(0, 1, 2)) / m

        self.W_fc -= self.lr * dW_fc
        self.b_fc -= self.lr * db_fc
        self.kernels -= self.lr * d_kernels
        self.bias_conv -= self.lr * d_bias_conv

        logging.info("CNN backward pass completed.")

    def train(self, train_loader, epochs=5, lr=0.01):
        self.lr = lr
        logging.info(f"Starting CNN training: epochs={epochs}, learning_rate={lr}")
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader()):
                output = self.forward(X_batch)
                loss = cross_entropy_loss(output, y_batch)
                total_loss += loss
                self.backward(y_batch)
                if batch_idx % 10 == 0:
                    logging.info(f"Epoch {epoch+1} Batch {batch_idx+1} Loss: {loss:.6f}")
            print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss:.4f}")

    def evaluate(self, test_loader):
        correct, total = 0, 0
        for X_batch, y_batch in test_loader():
            preds = np.argmax(self.forward(X_batch), axis=1)
            labels = np.argmax(y_batch, axis=1)
            correct += (preds == labels).sum()
            total += len(y_batch)
        print(f"Test Accuracy: {correct / total * 100:.2f}%")

def to_numpy_loader(loader):
    def generator():
        for images, labels in loader:
            images = images.squeeze().numpy()
            labels = one_hot(labels.numpy())
            yield images, labels
    return generator

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader_pt = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader_pt = DataLoader(test_dataset, batch_size=128, shuffle=False)

    train_loader = to_numpy_loader(train_loader_pt)
    test_loader = to_numpy_loader(test_loader_pt)

    model = CNN()
    model.train(train_loader, epochs=5, lr=0.01)
    model.evaluate(test_loader)

if __name__ == '__main__':
    main()
