import tensorflow as tf
import inspect

class Backbone_test:
    IMG_SHAPE = (224, 224, 3)
    input1 = tf.keras.Input(shape=IMG_SHAPE)
    base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                  include_top=True)
    # entire pretrained model
    tf.keras.utils.plot_model(model=base_model, to_file='base_model.png', show_shapes=True)
    # two methods to build test base_model2
    # base_model2 = tf.keras.Model(inputs=[base_model.layers[0].output], outputs=[base_model.layers[-3].output])
    base_model2 = tf.keras.Model(inputs=[base_model.get_layer(index=0).input], outputs=[base_model.get_layer(index=-3).output])
    tf.keras.utils.plot_model(model=base_model2, to_file='base_model_cut.png', show_shapes=True)
    # To build base_model3, we need input layer
    input2 = tf.keras.Input(shape=(7,7,2048))
    base_model3 = tf.keras.Model(inputs=[input2], outputs=[base_model.get_layer(index=-2)(input2)])
    tf.keras.utils.plot_model(model=base_model3, to_file='base_model_cut2.png', show_shapes=True)
    # better use Sequential API
    base_model4 = tf.keras.Sequential(layers=[
        input2,
        base_model.get_layer(index=-2),
        base_model.get_layer(index=-1)
    ])
    # Check if the weights are same in two models
    print(base_model.layers[-1].get_weights()[0].flatten()[:5])
    print(base_model4.layers[-1].get_weights()[0].flatten()[:5])
    tf.keras.utils.plot_model(model=base_model4, to_file='base_model_cut3.png', show_shapes=True)
    # print(base_model.summary())
    conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), padding='same')
    bh1 = tf.keras.layers.BatchNormalization()
    ac1 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)
    model2 = tf.keras.Model(inputs=[input1], outputs=[ac1(bh1(conv1(base_model2(input1))))])   # try functional API
    # Try Sequential API, better use Sequential API
    # model2 = tf.keras.Sequential(layers=[
    #     input1,
    #     base_model,
    #     conv1,
    #     bh1,
    #     ac1
    # ])
    tf.keras.utils.plot_model(model=model2, to_file='base_model_modified.png', show_shapes=True)

    print(len(base_model.layers))




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
    t1 = Backbone_test()
    # t1.plot_model()
    # print(t1.get_output_shape())



