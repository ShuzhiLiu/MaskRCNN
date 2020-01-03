import tensorflow as tf



class RPN:
    def __init__(self, INPUT_SHAPE=(23,40,256)):
        self.input1 = tf.keras.Input(shape=INPUT_SHAPE)
        self.conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same')(self.input1)
        self.bh1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.ac1 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(self.bh1)

        # decide foreground or background
        self.conv2 = tf.keras.layers.Conv2D(filters=18, kernel_size=(1,1), padding='same')(self.ac1)
        self.bh2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.ac2 = tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='foreground')(self.bh2)

        # bounding box regression
        self.conv3 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1,1), padding='same')(self.ac1)
        self.bh3 = tf.keras.layers.BatchNormalization()(self.conv3)
        self.ac3 = tf.keras.layers.Activation(activation=tf.keras.activations.linear, name='regression')(self.bh3)

        self.RPN_base_model = tf.keras.Model(inputs=[self.input1], outputs=[self.ac2, self.ac3], name='RPN_base')


        tf.keras.utils.plot_model(model=self.RPN_base_model, to_file='RPN_base.png', show_shapes=True)

    def gen_RPN_output_model(self):
        print(self.RPN_base_model.get_layer(name='foreground').output_shape)
        print(self.RPN_base_model.get_layer(name='regression').output_shape)
        shape_temp1 = self.RPN_base_model.get_layer(name='foreground').output_shape
        shape_temp2 = self.RPN_base_model.get_layer(name='regression').output_shape
        self.reshape2 = tf.keras.layers.Reshape(target_shape=(shape_temp1[1],shape_temp1[2],int(shape_temp1[3]/2),2))(self.ac2)
        self.output2= tf.nn.softmax(logits=self.reshape2, axis=-1, name='foreground2')

        self.reshape3 = tf.keras.layers.Reshape(target_shape=(shape_temp2[1],shape_temp2[2],int(shape_temp2[3]/4),4), name='regression2')(self.ac3)

        self.RPN_output_model = tf.keras.Model(inputs=[self.input1], outputs=[self.output2, self.reshape3])

        tf.keras.utils.plot_model(model=self.RPN_output_model, to_file='RPN_output.png', show_shapes=True)
        return self.RPN_output_model





if __name__=='__main__':
    t1 = RPN()
    t1.gen_RPN_output_model()
