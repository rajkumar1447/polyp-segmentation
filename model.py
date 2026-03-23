# Import Necessary Libraries
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.applications import ResNet50

# Function to add the conolutional Initial Layers.
def conv_block(x, filters):
    x = L.Conv2D(filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Conv2D(filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)

    x = L.Dropout(0.3)(x)
    return x

# Function to add the Backbone Main model. [UNet3+]
def unet3plus(input_shape):
    inputs = L.Input(input_shape)

    encoder = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    e1 = encoder.get_layer("input_1").output
    e2 = encoder.get_layer("conv1_relu").output
    e3 = encoder.get_layer("conv2_block3_out").output
    e4 = encoder.get_layer("conv3_block4_out").output
    e5 = encoder.get_layer("conv4_block6_out").output

    def up(x, scale): return L.UpSampling2D((scale, scale), interpolation="bilinear")(x)
    def down(x, scale): return L.MaxPool2D((scale, scale))(x)

    def fuse(e1,e2,e3,e4,e5):
        return L.Concatenate()([
            conv_block(e1,64),
            conv_block(e2,64),
            conv_block(e3,64),
            conv_block(e4,64),
            conv_block(e5,64)
        ])

    d4 = fuse(down(e1,8), down(e2,4), down(e3,2), e4, up(e5,2))
    d3 = fuse(down(e1,4), down(e2,2), e3, up(d4,2), up(e5,4))
    d2 = fuse(down(e1,2), e2, up(d3,2), up(d4,4), up(e5,8))
    d1 = fuse(e1, up(d2,2), up(d3,4), up(d4,8), up(e5,16))

    output = L.Conv2D(1, 1, activation="sigmoid")(d1)

    return tf.keras.Model(inputs, output)