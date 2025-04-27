import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from utils.activations import sigmoid, softmax
from utils.loss import cross_entropy_loss

class MLP:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y, output):
        m = X.shape[0]

        dZ2 = output - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, train_loader, epochs=10, lr=0.1):
        self.lr = lr
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                output = self.forward(X_batch)
                loss = cross_entropy_loss(output, y_batch)
                total_loss += loss
                self.backward(X_batch, y_batch, output)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, test_loader):
        correct, total = 0, 0
        for X_batch, y_batch in test_loader:
            preds = self.predict(X_batch)
            labels = np.argmax(y_batch, axis=1)
            correct += (preds == labels).sum()
            total += len(y_batch)
        print(f"Test Accuracy: {correct / total * 100:.2f}%")
