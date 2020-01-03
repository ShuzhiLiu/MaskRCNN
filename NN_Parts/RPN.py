import tensorflow as tf
from NN_Parts import Backbone



class RPN:
    def __init__(self, backbone_model):
        back_outshape = backbone_model.layers[-1].output.shape[1:]
        self.conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', name='RPN_START')(backbone_model.layers[-1].output)
        self.bh1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.ac1 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(self.bh1)

        # decide foreground or background
        self.conv2 = tf.keras.layers.Conv2D(filters=18, kernel_size=(1,1), padding='same')(self.ac1)
        self.bh2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.ac2 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(self.bh2)
        self.reshape2 = tf.keras.layers.Reshape(
            target_shape=(back_outshape[0], back_outshape[1], int(18 /2 ), 2))(self.ac2)
        self.output2 = tf.nn.softmax(logits=self.reshape2, axis=-1, name='RPN_Anchor_Pred')

        # bounding box regression
        self.conv3 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1,1), padding='same')(self.ac1)
        self.bh3 = tf.keras.layers.BatchNormalization()(self.conv3)
        self.ac3 = tf.keras.layers.Activation(activation=tf.keras.activations.linear)(self.bh3)
        self.reshape3 = tf.keras.layers.Reshape(
            target_shape=(back_outshape[0], back_outshape[1], int(36 / 4), 4), name='RPN_BBOX_Regression')(self.ac3)

        self.RPN_model = tf.keras.Model(inputs=[backbone_model.layers[0].input], outputs=[self.output2, self.reshape3], name='RPN_model')


        tf.keras.utils.plot_model(model=self.RPN_model, to_file='RPN_with_backbone.png', show_shapes=True)


    def _RPN_loss(self):
        pass









if __name__=='__main__':
    b1 = Backbone()
    t1 = RPN(b1.backbone_model)
