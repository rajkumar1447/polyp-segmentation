# Import Necessary Libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Global Variables
IMG_H, IMG_W = 256, 256

# Function to Read the Images and resize.
def read_image(path):
    path = path.decode()
    image = cv2.imread(path)
    image = cv2.resize(image, (IMG_W, IMG_H))
    image = preprocess_input(image)
    return image.astype(np.float32)

# Function to Read the corresponding mask.
def read_mask(path):
    path = path.decode()
    mask = cv2.imread(path, 0)
    mask = cv2.resize(mask, (IMG_W, IMG_H))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask.astype(np.float32)

# Function to do the augment
def augment(x, y):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)

    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)

    if tf.random.uniform(()) > 0.5:
        x = tf.image.random_brightness(x, 0.2)

    if tf.random.uniform(()) > 0.5:
        x = tf.image.random_contrast(x, 0.8, 1.2)

    if tf.random.uniform(()) > 0.5:
        x = tf.image.rot90(x)
        y = tf.image.rot90(y)

    return x, y

# Function to parse the data flow.
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMG_H, IMG_W, 3])
    y.set_shape([IMG_H, IMG_W, 1])
    return x, y

# Function to prepare the dataset based on the tf
def tf_dataset(X, Y, batch):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(100)

    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds