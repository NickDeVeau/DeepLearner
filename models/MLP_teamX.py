import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.activations import sigmoid, softmax
from utils.loss import cross_entropy_loss

# Setup logger
logging.basicConfig(filename='mlp_training_log.txt', level=logging.INFO, filemode='w')
logger = logging.getLogger()

class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        logger.info(f"Initializing MLP model: input {input_size}, hidden {hidden_size}, output {output_size}")
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        logger.info(f"MLP forward pass: input shape {X.shape}")
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        logger.info(f"MLP forward pass completed: output shape {self.A2.shape}")
        return self.A2

    def backward(self, X, y, output):
        logger.info(f"MLP backward pass: input {X.shape}, output {output.shape}")
        m = X.shape[0]
        dZ2 = output - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        logger.info(f"MLP backward pass completed.")

    def train(self, loader_fn, epochs=10, lr=0.1):
        self.lr = lr
        logger.info(f"Starting training: epochs={epochs}, learning_rate={lr}")
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (X_batch, y_batch) in enumerate(loader_fn()):
                output = self.forward(X_batch)
                loss = cross_entropy_loss(output, y_batch)
                total_loss += loss
                self.backward(X_batch, y_batch, output)
                logger.info(f"Epoch {epoch+1} Batch {batch_idx+1} Loss: {loss:.6f}")
            print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss:.4f}")

    def predict(self, X):
        logger.info(f"MLP prediction on input shape {X.shape}")
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, loader):
        correct = total = 0
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            preds = self.predict(X_batch)
            labels = np.argmax(y_batch, axis=1)
            correct += (preds == labels).sum()
            total += len(y_batch)
            logger.info(f"Evaluating batch {batch_idx+1}")
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"Final Test Accuracy: {accuracy * 100:.2f}%")

def one_hot(labels, num_classes=10):
    return np.eye(num_classes, dtype=np.float32)[labels]

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_ds = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader_pt = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader_pt = DataLoader(test_ds, batch_size=128, shuffle=False)

    def to_numpy_loader(pt_loader):
        for imgs, labels in pt_loader:
            yield imgs.view(imgs.size(0), -1).numpy(), one_hot(labels.numpy())

    model = MLP()
    model.train(lambda: to_numpy_loader(train_loader_pt), epochs=10, lr=0.1)
    model.evaluate(to_numpy_loader(test_loader_pt))

if __name__ == '__main__':
    main()
