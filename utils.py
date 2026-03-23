# Import Necessary Libraries.
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create Directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Seed Everything.
def set_seed(seed=42):
    import random
    import tensorflow as tf

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Save Visualization
def save_results(image, mask, pred, save_path):
    mask = np.repeat(mask, 3, axis=-1)
    pred = np.repeat(pred, 3, axis=-1)

    line = np.ones((image.shape[0], 10, 3)) * 255

    combined = np.concatenate([image, line, mask*255, line, pred*255], axis=1)

    cv2.imwrite(save_path, combined)

# Plot Training History
def plot_history(history):
    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss")

    # Dice
    plt.subplot(1,2,2)
    plt.plot(history.history['dice_coef'], label='train_dice')
    plt.plot(history.history['val_dice_coef'], label='val_dice')
    plt.legend()
    plt.title("Dice Score")

    plt.show()

# Threshold Prediction
def threshold_mask(pred, threshold=0.5):
    return (pred > threshold).astype(np.uint8)