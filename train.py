# Import Necessary Libraries
import os
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from dataset import tf_dataset
from model import build_model
from metrics import combined_loss, dice_coef, iou
from utils import set_seed, plot_history

set_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Function to create Dir
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Global
BATCH = 8
EPOCHS = 30

image_paths = sorted(glob("data/images/*"))
mask_paths = sorted(glob("data/masks/*"))

train_x, val_x, train_y, val_y = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

train_ds = tf_dataset(train_x, train_y, BATCH)
val_ds = tf_dataset(val_x, val_y, BATCH)

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss,
    metrics=[dice_coef, iou]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("files/model.h5", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.CSVLogger("files/log.csv")
]

# Training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Plot Results
plot_history(history)