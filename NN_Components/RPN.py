import numpy as np
import tensorflow as tf

from NN_Components import Backbone
from NN_Helper import BboxTools


class RPN:
    def __init__(self,
                 backbone_model,
                 lambda_factor: int = 1,
                 batch: int = 1,
                 lr: float = 1e-4):
        self.LAMBDA_FACTOR = lambda_factor
        self.BATCH = batch
        self.lr = lr
        # the part of backbone
        self.backbone_model = backbone_model
        back_outshape = backbone_model.output.shape[1:]
        self.input_backbone = tf.keras.Input(shape=backbone_model.input.shape[1:], name='BACKBONE_INPUT',
                                             dtype=tf.float32)

        self.input_RPN = tf.keras.Input(shape=back_outshape, batch_size=None, name='RPN_INPUT', dtype=tf.float32)
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

        self.RPN_header_model = tf.keras.Model(inputs=[self.input_RPN],
                                               outputs=[self.RPN_Anchor_Pred, self.RPN_BBOX_Regression_Pred],
                                               name='RPN_HEADER_MODEL')
        backbone_out = backbone_model(self.input_backbone)
        rpn_anchor_pred, rpn_bbox_regression_pred = self.RPN_header_model(backbone_out)
        self.RPN_with_backbone_model = tf.keras.Model(inputs=[self.input_backbone],
                                                      outputs=[rpn_anchor_pred, rpn_bbox_regression_pred],
                                                      name='RPN_BACKBONE_MODEL')

        self.shape_Anchor_Target = self.RPN_header_model.get_layer(
            name='tf_op_layer_RPN_Anchor_Pred').output.shape[1:-1]
        self.shape_BBOX_Regression = self.RPN_header_model.get_layer(
            name='RPN_BBOX_Regression_Pred').output.shape[1:]
        self.N_total_anchors = self.shape_Anchor_Target[0] * self.shape_Anchor_Target[1] * self.shape_Anchor_Target[2]

        # --- for low level training ---
        self.optimizer_with_backbone = tf.keras.optimizers.Adam(self.lr)
        self.optimizer_header = tf.keras.optimizers.Adam(self.lr)

    def process_image(self, img):
        rpn_anchor_pred, rpn_bbox_regression_pred = self.RPN_with_backbone_model.predict(img)
        return rpn_anchor_pred, rpn_bbox_regression_pred

    def save_model(self, root_path: str):
        self.RPN_with_backbone_model.save_weights(filepath=f"{root_path}/RPN_model")

    def load_model(self, root_path: str):
        self.RPN_with_backbone_model.load_weights(filepath=f"{root_path}/RPN_model")

    def plot_model(self):
        tf.keras.utils.plot_model(model=self.RPN_header_model, to_file='RPN_header_model.png', show_shapes=True)
        tf.keras.utils.plot_model(model=self.RPN_with_backbone_model, to_file='RPN_with_backbone.png', show_shapes=True)

        self._rpn_train_model()

    def _rpn_train_model(self):
        self.RPN_Anchor_Target = tf.keras.Input(shape=self.shape_Anchor_Target, name='RPN_Anchor_Target')
        self.RPN_BBOX_Regression_Target = tf.keras.Input(shape=self.shape_BBOX_Regression,
                                                         name='RPN_BBOX_Regression_Target')
        self.RPN_train_model = tf.keras.Model(inputs=[self.RPN_with_backbone_model.inputs,
                                                      self.RPN_Anchor_Target,
                                                      self.RPN_BBOX_Regression_Target],
                                              outputs=[self.RPN_with_backbone_model.outputs],
                                              name='RPN_train_model')
        self.RPN_train_model.add_loss(losses=self._rpn_loss(anchor_target=self.RPN_Anchor_Target,
                                                            bbox_reg_target=self.RPN_BBOX_Regression_Target,
                                                            anchor_pred=self.RPN_with_backbone_model.outputs[0],
                                                            bbox_reg_pred=self.RPN_with_backbone_model.outputs[
                                                                1]))  # Always check if the layers are in current graph!
        self.RPN_train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

        tf.keras.utils.plot_model(model=self.RPN_train_model, to_file='RPN_train_model.png', show_shapes=True)

    def _rpn_loss(self, anchor_target, bbox_reg_target, anchor_pred, bbox_reg_pred):
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
        huber_loss = tf.losses.Huber()
        bbox_reg_loss = huber_loss(y_true=bbox_reg_target, y_pred=bbox_reg_pred)
        bbox_reg_loss = tf.math.reduce_mean(bbox_reg_loss)
        bbox_reg_loss = tf.math.multiply(bbox_reg_loss, self.LAMBDA_FACTOR)
        total_loss = tf.add(anchor_loss, bbox_reg_loss)

        return total_loss

    def _proposal_boxes(self,
                        rpn_anchor_pred,
                        rpn_bbox_regression_pred,
                        anchor_candidates,
                        h: int,
                        w: int,
                        n_anchors: int,
                        n_proposal: int,
                        anchor_threshold: float):
        # === Selection part ===
        # top_values, top_indices = tf.math.top_k()
        rpn_anchor_pred = tf.slice(rpn_anchor_pred, [0, 0, 0, 0, 1], [1,
                                                                      h,
                                                                      w,
                                                                      n_anchors,
                                                                      1])  # second channel is foreground
        # squeeze the pred of anchor and bbox_reg
        rpn_anchor_pred = tf.squeeze(rpn_anchor_pred)
        rpn_bbox_regression_pred = tf.squeeze(rpn_bbox_regression_pred)
        shape1 = tf.shape(rpn_anchor_pred)
        # flatten the pred of anchor to get top N values and indices
        rpn_anchor_pred = tf.reshape(rpn_anchor_pred, (-1,))
        n_anchor_proposal = n_proposal
        top_values, top_indices = tf.math.top_k(rpn_anchor_pred,
                                                n_anchor_proposal)  # top_k has sort function. it's important here
        top_indices = tf.gather_nd(top_indices, tf.where(tf.greater(top_values, anchor_threshold)))
        top_values = tf.gather_nd(top_values, tf.where(tf.greater(top_values, anchor_threshold)))

        top_indices = tf.reshape(top_indices, (-1, 1))
        update_value = tf.math.add(top_values, 1)
        rpn_anchor_pred = tf.tensor_scatter_nd_update(rpn_anchor_pred, top_indices, update_value)
        rpn_anchor_pred = tf.reshape(rpn_anchor_pred, shape1)

        # --- find the base boxes ---
        anchor_pred_top_indices = tf.where(tf.greater(rpn_anchor_pred, 1))
        base_boxes = tf.gather_nd(anchor_candidates, anchor_pred_top_indices)

        # --- find the bbox_regs ---
        # flatten the bbox_reg by last dim to use top_indices to get final_box_reg
        rpn_bbox_regression_pred_shape = tf.shape(rpn_bbox_regression_pred)
        rpn_bbox_regression_pred = tf.reshape(rpn_bbox_regression_pred, (-1, rpn_bbox_regression_pred_shape[-1]))
        final_box_reg = tf.gather_nd(rpn_bbox_regression_pred, top_indices)

        # Convert to numpy to plot
        final_box = BboxTools.bbox_reg2truebox(base_boxes=base_boxes, regs=final_box_reg)
        return np.array(final_box).astype(np.float)
        # return final_box

    @tf.function
    def train_step_with_backbone(self, image, anchor_target, box_reg_target):
        with tf.GradientTape() as backbone_tape:
            anchor_pred, box_reg_pred = self.RPN_with_backbone_model(image)
            total_loss = self._rpn_loss(anchor_target, box_reg_target, anchor_pred, box_reg_pred)
        gradients_backbone = backbone_tape.gradient(total_loss, self.RPN_with_backbone_model.trainable_variables)
        self.optimizer_with_backbone.apply_gradients(
            zip(gradients_backbone, self.RPN_with_backbone_model.trainable_variables))

    @tf.function
    def train_step_header(self, image, anchor_target, box_reg_target):
        with tf.GradientTape() as header_tape:
            anchor_pred, box_reg_pred = self.RPN_with_backbone_model(image)
            total_loss = self._rpn_loss(anchor_target, box_reg_target, anchor_pred, box_reg_pred)
        gradients_header = header_tape.gradient(total_loss, self.RPN_header_model.trainable_variables)
        self.optimizer_header.apply_gradients(zip(gradients_header, self.RPN_header_model.trainable_variables))


if __name__ == '__main__':
    b1 = Backbone()
    t1 = RPN(b1.backbone_model)
    t1.plot_model()
