import tensorflow as tf



class RPN:
    def __init__(self, INPUT_SHAPE=(40,23,256)):
        self.input1 = tf.keras.Input(shape=INPUT_SHAPE)
        self.conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same')(self.input1)
        self.bh1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.ac1 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(self.bh1)
        
