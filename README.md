# Deep Learning Project: Manual MLP and CNN Implementation

This repository contains manual implementations of:

- A **Multi-Layer Perceptron (MLP)** using fully connected layers
- A **Convolutional Neural Network (CNN)** with manual 2D convolution

Both models are trained and evaluated on the MNIST dataset.  
Only **NumPy** is used for model logic.  
**Torch** and **torchvision** are used exclusively for loading and batching the dataset.

---

## Directory Structure

```
data/                # MNIST data (downloaded automatically)
models/
 ├── MLP_teamX.py    # MLP model implementation
 └── CNN_teamX.py    # CNN model implementation
utils/
 ├── activations.py  # Sigmoid, Softmax activation functions
 └── loss.py         # Cross-entropy loss function
README.md
requirements.txt
```

---

## Environment Setup

1. Create a Conda environment and activate it:

```bash
conda create -n dl_proj python=3.9
conda activate dl_proj
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

---

## How to Train and Evaluate

### MLP

Run:

```bash
python models/MLP_teamX.py
```

- Loads MNIST dataset.
- Trains the MLP model for 10 epochs.
- Prints training loss per epoch.
- Prints final test accuracy.

### CNN

Run:

```bash
python models/CNN_teamX.py
```

- Loads MNIST dataset.
- Trains the CNN model for 5 epochs.
- Prints batch loss during training.
- Prints final test accuracy.

---

## Testing without Retraining

These scripts are designed to **train from scratch** every time.  
To test without retraining:
- Modify the `main()` function in each file to skip `train()` and only call `evaluate()`.

---

## Additional Notes

- Training is executed locally on CPU (macOS).
- No GPU acceleration is used.
- Runtime is dependent on hardware; training may take a few minutes on CPU.
- All model computations (forward and backward) are implemented manually using NumPy.

---

## Requirements

See `requirements.txt` for package versions.  
Minimal set:

```
numpy>=1.21.0
torch>=1.9.0
torchvision>=0.10.0
```

