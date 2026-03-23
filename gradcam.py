# Import Necessary Libraries.
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function for the gradcam.
def gradcam(model, img):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.layers[-3].output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img)
        loss = tf.reduce_mean(pred)

    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    cam = np.dot(conv_out[0], weights)

    cam = np.maximum(cam,0)
    cam = cam / (cam.max() + 1e-8)

    return cam