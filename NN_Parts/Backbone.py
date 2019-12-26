import tensorflow as tf
import inspect

class Backbone_test:
    IMG_SHAPE = (1280, 720, 3)
    input1 = tf.keras.Input(shape=IMG_SHAPE)
    base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                  include_top=False)
    tf.keras.utils.plot_model(model=base_model, to_file='base_model.png', show_shapes=True)
    # print(base_model.summary())
    conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same')
    bh1 = tf.keras.layers.BatchNormalization()
    ac1 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)
    # model2 = tf.keras.Model(inputs=[input1], outputs=[ac1(bh1(conv1(base_model(input1))))])
    model2 = tf.keras.Sequential(layers=[
        input1,
        base_model,
        conv1,
        bh1,
        ac1
    ])
    tf.keras.utils.plot_model(model=model2, to_file='base_model_modified.png', show_shapes=True)




class Backbone:
    def __init__(self, IMG_SHAPE=(1280,720,3)):
        # TODO: change stages to 4, current is 5
        self.base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                           include_top = False)
        conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same')
        bh1 = tf.keras.layers.BatchNormalization()
        ac1 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)
        self.model_modified = tf.keras.Sequential(layers=[
            self.base_model,
            conv1,
            bh1,
            ac1
        ])

    def plot_model(self):
        tf.keras.utils.plot_model(model=self.model_modified, to_file='base_model_modified.png', show_shapes=True)

    def get_output_shape(self):
        return self.model_modified.layers[-1].output_shape[1:]  # first dim is batch size

    def save_weight(self):
        self.model_modified.save_weights(filepath='SavedWeights/Backbone.ckpt')
    def load_weight(self):
        self.model_modified.load_weights(filepath='SavedWeights/Backbone.ckpt')



if __name__=='__main__':
    t1 = Backbone()
    t1.plot_model()
    print(t1.get_output_shape())



