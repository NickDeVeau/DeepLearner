import numpy as np
import logging

logger = logging.getLogger()

def sigmoid(x):
    logger.info(f"Calling sigmoid activation: input shape {x.shape}")
    output = 1 / (1 + np.exp(-x))
    logger.info(f"Sigmoid activation completed: output shape {output.shape}")
    return output

def relu(x):
    logger.info(f"Calling ReLU activation: input shape {x.shape}")
    output = np.maximum(0, x)
    logger.info(f"ReLU activation completed: output shape {output.shape}")
    return output

def softmax(x):
    logger.info(f"Calling softmax activation: input shape {x.shape}")
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    output = exps / np.sum(exps, axis=1, keepdims=True)
    logger.info(f"Softmax activation completed: output shape {output.shape}")
    return output
