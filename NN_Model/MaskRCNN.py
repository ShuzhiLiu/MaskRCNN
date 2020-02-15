import tensorflow as tf
from NN_Components import Backbone

class MaskRCNN():
    def __init__(self,IMG_SHAPE):
        self.input = tf.keras.Input(shape=IMG_SHAPE)
        self.backbone = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top = False)
