# Import Necessary Libraries.
import os, cv2, numpy as np
import tensorflow as tf
from glob import glob
from metrics import dice_coef, iou, bce_dice_loss

# Model
model = tf.keras.models.load_model(
    "files/model.h5",
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "iou": iou
    }
)
images = sorted(glob("data/images/*"))
masks = sorted(glob("data/masks/*"))

dice_scores, iou_scores = [], []

for x, y in zip(images, masks):
    img = cv2.imread(x)
    img = cv2.resize(img,(512,512))/255.0
    pred = model.predict(np.expand_dims(img,0))[0]
    pred = (pred > 0.5).astype(np.float32)

    gt = cv2.imread(y,0)
    gt = cv2.resize(gt,(512,512))/255.0
    gt = np.expand_dims(gt,-1)

    d = dice_coef(gt, pred).numpy()
    j = iou(gt, pred).numpy()

    dice_scores.append(d)
    iou_scores.append(j)

print("Dice:", np.mean(dice_scores))
print("IoU:", np.mean(iou_scores))