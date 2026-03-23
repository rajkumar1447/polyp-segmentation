# Import Necessary Libraries
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

# Function to do the decoding.
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

# Function to add the conolutional Initial Layers.
def attention_block(x, g, filters):
    theta_x = layers.Conv2D(filters, 1, padding="same")(x)
    phi_g = layers.Conv2D(filters, 1, padding="same")(g)

    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation("relu")(add)

    psi = layers.Conv2D(1, 1, padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)

    out = layers.Multiply()([x, psi])
    return out

# Function to add the Backbone Main model. [UNet3+]
def decoder_block(x, skip, filters):
    x = layers.UpSampling2D((2,2))(x)

    if skip is not None:
        skip = attention_block(skip, x, filters)
        x = layers.Concatenate()([x, skip])

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x

# Function to build model
def build_model():
    inputs = layers.Input((256, 256, 3))

    base = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    s1 = base.get_layer("conv1_relu").output
    s2 = base.get_layer("conv2_block3_out").output
    s3 = base.get_layer("conv3_block4_out").output
    s4 = base.get_layer("conv4_block6_out").output
    b1 = base.get_layer("conv5_block3_out").output

    # Decoder with dense skip connections
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Extra refinement layer (important improvement)
    d5 = conv_block(layers.UpSampling2D((2,2))(d4), 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d5)

    return Model(inputs, outputs)