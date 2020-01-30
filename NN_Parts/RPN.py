import tensorflow as tf
from NN_Parts import Backbone
import random
import numpy as np
from Debugger import DebugPrint


class RPN:
    def __init__(self, backbone_model, lambda_factor=1, batch=1, lr=1e-4):
        self.LAMBDA_FACTOR = lambda_factor
        self.BATCH = batch
        self.lr = lr
        # the part of backbone
        back_outshape = backbone_model.output.shape[1:]

        self.input_RPN = tf.keras.Input(shape=back_outshape,batch_size=None, name='RPN_INPUT', dtype=tf.float32)
        self.conv1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                                            kernel_initializer='he_normal', name='RPN_START')(
            self.input_RPN)
        self.bh1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.ac1 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(self.bh1)

        # decide foreground or background
        self.conv2 = tf.keras.layers.Conv2D(filters=18, kernel_size=(1, 1), padding='same',
                                            kernel_initializer='he_normal')(self.ac1)
        self.bh2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.ac2 = tf.keras.layers.Activation(activation=tf.keras.activations.relu)(self.bh2)
        self.reshape2 = tf.keras.layers.Reshape(
            target_shape=(back_outshape[0], back_outshape[1], int(18 / 2), 2))(self.ac2)
        self.RPN_Anchor_Pred = tf.nn.softmax(logits=self.reshape2, axis=-1, name='RPN_Anchor_Pred')
        # [1,0] is background, [0,1] is foreground. second channel is true.

        # bounding box regression
        self.conv3 = tf.keras.layers.Conv2D(filters=36, kernel_size=(1, 1), padding='same',
                                            kernel_initializer='he_normal')(self.ac1)
        self.bh3 = tf.keras.layers.BatchNormalization()(self.conv3)
        self.ac3 = tf.keras.layers.Activation(activation=tf.keras.activations.linear)(self.bh3)
        self.RPN_BBOX_Regression_Pred = tf.keras.layers.Reshape(
            target_shape=(back_outshape[0], back_outshape[1], int(36 / 4), 4), name='RPN_BBOX_Regression_Pred')(
            self.ac3)

        self.RPN_header_model = tf.keras.Model(inputs=[self.input_RPN],outputs=[self.RPN_Anchor_Pred, self.RPN_BBOX_Regression_Pred])
        RPN_Anchor_Pred, RPN_BBOX_Regression_Pred = self.RPN_header_model([backbone_model.output])
        self.RPN_with_backbone_model = tf.keras.Model(inputs=[backbone_model.inputs],
                                                      outputs=[RPN_Anchor_Pred, RPN_BBOX_Regression_Pred],
                                                      name='RPN_BACKBONE_MODEL')


        self.shape_Anchor_Target = self.RPN_header_model.get_layer(
            name='tf_op_layer_RPN_Anchor_Pred').output.shape[1:-1]
        self.shape_BBOX_Regression = self.RPN_header_model.get_layer(
            name='RPN_BBOX_Regression_Pred').output.shape[1:]
        self.N_total_anchors = self.shape_Anchor_Target[0] * self.shape_Anchor_Target[1] * self.shape_Anchor_Target[2]

        # tf.keras.utils.plot_model(model=self.RPN_header_model, to_file='RPN_header_model.png', show_shapes=True)
        tf.keras.utils.plot_model(model=self.RPN_with_backbone_model, to_file='RPN_with_backbone.png', show_shapes=True)

        # self._RPN_train_model()

        # --- for low level training ---
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _RPN_train_model(self):
        self.RPN_Anchor_Target = tf.keras.Input(shape=self.shape_Anchor_Target, name='RPN_Anchor_Target')
        self.RPN_BBOX_Regression_Target = tf.keras.Input(shape=self.shape_BBOX_Regression,
                                                         name='RPN_BBOX_Regression_Target')
        self.RPN_train_model = tf.keras.Model(inputs=[self.RPN_with_backbone_model.inputs,
                                                      self.RPN_Anchor_Target,
                                                      self.RPN_BBOX_Regression_Target],
                                              outputs=[self.RPN_with_backbone_model.outputs],
                                              name='RPN_train_model')
        self.RPN_train_model.add_loss(losses=self._RPN_loss(anchor_target=self.RPN_Anchor_Target,
                                                            bbox_reg_target=self.RPN_BBOX_Regression_Target,
                                                            anchor_pred=self.RPN_Anchor_Pred,
                                                            bbox_reg_pred=self.RPN_BBOX_Regression_Pred))
        self.RPN_train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

        tf.keras.utils.plot_model(model=self.RPN_train_model, to_file='RPN_train_model.png', show_shapes=True)

    def _RPN_loss(self, anchor_target, bbox_reg_target, anchor_pred, bbox_reg_pred):
        # shape of input anchor_target: (batch_size=1, h, w, n_anchors)
        # currently only support batch size = 1
        bbox_inside_weight = tf.zeros(
            shape=(self.BATCH, self.shape_Anchor_Target[0], self.shape_Anchor_Target[1], self.shape_Anchor_Target[2]),
            dtype=tf.float32)

        # --- anchor_target: 1:foreground, 0.5:ignore, 0:background ---
        indices_foreground = tf.where(tf.equal(anchor_target, 1))
        indices_background = tf.where(tf.equal(anchor_target, 0))
        # DebugPrint('indices_foreground', indices_foreground)
        # DebugPrint('indices_background', indices_background)
        n_foreground = tf.gather_nd(tf.shape(indices_foreground), [[0]])
        # DebugPrint('n_foreground', n_foreground)

        # --- update value of bbox_inside_weight corresponding to foreground to 1 ---
        bbox_inside_weight = tf.tensor_scatter_nd_update(tensor=bbox_inside_weight, indices=indices_foreground,
                                                         updates=tf.ones(shape=n_foreground))

        # --- balance the foreground and background training sample ---
        n_background_selected = 128
        selected_ratio = (n_foreground + n_background_selected) / self.N_total_anchors
        remain_ratio = (self.N_total_anchors - n_foreground - n_background_selected) / self.N_total_anchors
        concat = tf.concat([remain_ratio, selected_ratio], axis=0)
        concat = tf.reshape(concat, (1, 2))
        temp_random_choice = tf.random.categorical(tf.math.log(concat), self.N_total_anchors)
        temp_random_choice = tf.reshape(temp_random_choice, (
            self.BATCH, self.shape_Anchor_Target[0], self.shape_Anchor_Target[1], self.shape_Anchor_Target[2]))
        temp_random_choice = tf.gather_nd(params=temp_random_choice, indices=indices_background)
        temp_random_choice = tf.dtypes.cast(temp_random_choice, tf.float32)
        # --- update value of bbox_inside_weight corresponding to random selected background to 1 ---
        bbox_inside_weight = tf.tensor_scatter_nd_update(tensor=bbox_inside_weight, indices=indices_background,
                                                         updates=temp_random_choice)

        indices_train = tf.where(tf.equal(bbox_inside_weight, 1))

        # print(anchor_target)
        # print(anchor_pred)
        # train anchor for foreground and background
        anchor_target = tf.cast(anchor_target, tf.int32)
        anchor_target = tf.one_hot(indices=anchor_target, depth=2, axis=-1)
        anchor_target = tf.gather_nd(params=anchor_target, indices=indices_train)
        anchor_pred = tf.gather_nd(params=anchor_pred, indices=indices_train)
        # --- train bbox reg only for foreground ---
        bbox_reg_target = tf.gather_nd(params=bbox_reg_target, indices=indices_foreground)
        bbox_reg_pred = tf.gather_nd(params=bbox_reg_pred, indices=indices_foreground)

        anchor_loss = tf.losses.categorical_crossentropy(y_true=anchor_target, y_pred=anchor_pred)
        anchor_loss = tf.math.reduce_mean(anchor_loss)
        Huberloss = tf.losses.Huber()
        bbox_reg_loss = Huberloss(y_true=bbox_reg_target, y_pred=bbox_reg_pred)
        bbox_reg_loss = tf.math.reduce_mean(bbox_reg_loss)
        bbox_reg_loss = tf.math.multiply(bbox_reg_loss, self.LAMBDA_FACTOR)
        total_loss = tf.add(anchor_loss, bbox_reg_loss)

        return total_loss

    @tf.function
    def train_step_with_backbone(self, image, anchor_target, box_reg_target):
        with tf.GradientTape() as backbone_tape, tf.GradientTape() as header_tape:
            anchor_pred, box_reg_pred = self.RPN_with_backbone_model(image)
            total_loss = self._RPN_loss(anchor_target, box_reg_target, anchor_pred, box_reg_pred)
        gradients_backbone = backbone_tape.gradient(total_loss, self.RPN_with_backbone_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_backbone, self.RPN_with_backbone_model.trainable_variables))
        gradients_header = header_tape.gradient(total_loss, self.RPN_header_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_header, self.RPN_header_model.trainable_variables))



if __name__ == '__main__':
    b1 = Backbone()
    t1 = RPN(b1.backbone_model)
