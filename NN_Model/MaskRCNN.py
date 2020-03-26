import tensorflow as tf
from NN_Components import Backbone


class MaskRCNN():
    def __init__(self, img_shape):
        self.input = tf.keras.Input(shape=img_shape)
        self.backbone = tf.keras.applications.ResNet50V2(input_shape=img_shape, include_top=False)
