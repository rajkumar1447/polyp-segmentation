# Import Necessary Libraries
import cv2
import numpy as np
import tensorflow as tf

# Global Variables
IMG_H, IMG_W = 512, 512


# Function to Read the Images and resize.
def read_image(path):
    path = path.decode()
    image = cv2.imread(path)
    image = cv2.resize(image, (IMG_W, IMG_H))
    image = image / 255.0
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
    if np.random.rend() > 0.5:
        x = np.fliplr(x)
        y = np.fliplr(y)
    if np.random.rand() > 0.5:
        x = np.flipud(x)
        y = np.flipud(y)  
    return x, y

# Function to parse the data flow.
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        x, y = augment(x, y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMG_H, IMG_W, 3])
    y.set_shape([IMG_H, IMG_W, 1])
    return x, y


# Function to prepare the dataset based on the tf
def tf_dataset(X, Y, batch):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse)
    ds = ds.batch(batch).prefetch(10)
    return ds