# Import Necessary Libraries.
import tensorflow as tf
import numpy as np
import cv2

# Function for the gradcam.
def grad_cam(model, image, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = tf.reduce_sum(predictions)

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    cam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    return cv2.resize(cam.numpy(), (256, 256))