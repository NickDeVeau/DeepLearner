import numpy as np

def cross_entropy_loss(predictions, targets):
    m = predictions.shape[0]
    clipped_preds = np.clip(predictions, 1e-12, 1. - 1e-12)
    loss = -np.sum(targets * np.log(clipped_preds)) / m
    return loss
