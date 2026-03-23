#  Import Necessary Libraries
import tensorflow as tf

smooth = 1e-6

# Function to calculate the dice coefficient.
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )
    
# Function to calculate dice loss
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Function to calculate binary cross entropy dice loss
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    pt = tf.exp(-bce)
    return alpha * (1 - pt) ** gamma * bce

def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

# Function to calculate the IOU
def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)