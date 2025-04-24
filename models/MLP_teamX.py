import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def main():
    # Load MNIST using torchvision (only for loading data!)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Wrap PyTorch dataloader with NumPy conversion
    def to_numpy_loader(loader):
        for images, labels in loader:
            images = images.view(images.size(0), -1).numpy()      # [batch_size, 784]
            labels = one_hot(labels.numpy())                      # One-hot encoded
            yield images, labels

    model = MLP(input_size=784, hidden_size=128, output_size=10)
    model.train(to_numpy_loader(train_loader), epochs=10, lr=0.1)
    model.evaluate(to_numpy_loader(test_loader))

if __name__ == '__main__':
    main()
