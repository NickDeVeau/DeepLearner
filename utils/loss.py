import numpy as np
import logging

logger = logging.getLogger()

def cross_entropy_loss(predictions, targets):
    logger.info(f"Calling cross_entropy_loss: predictions shape {predictions.shape}, targets shape {targets.shape}")
    m = predictions.shape[0]
    clipped_preds = np.clip(predictions, 1e-12, 1. - 1e-12)
    loss = -np.sum(targets * np.log(clipped_preds)) / m
    logger.info(f"Cross entropy loss computed: {loss:.6f}")
    return loss
