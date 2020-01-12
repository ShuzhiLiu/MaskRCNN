from Debugger import DebugPrint
import numpy as np
import tensorflow as tf

class RoI:
    def __init__(self, IMG_SHAPE):
        image_input = tf.keras.Input(shape=IMG_SHAPE, name='IMAGE_INPUT')
        proposal_boxes = tf.keras.Input(shape=(4,),batch_size = None,name='PROPOSAL_BOXES')
        shape1 = tf.shape(proposal_boxes)
        n_boxes = tf.gather_nd(shape1, [0])
        indices = tf.ones(shape=n_boxes)
        image_crop = tf.image.crop_and_resize(image_input, proposal_boxes,indices, [7,7])

    def proposal(self,anchor_candidates):
        # === Prediction part ===
        inputs, anchor_targets, bbox_targets = self.train_data_generator.gen_train_data()
        print(inputs.shape, anchor_targets.shape, bbox_targets.shape)
        input1 = np.reshape(inputs[0, :, :, :], (1, 720, 1280, 3))
        RPN_Anchor_Pred, RPN_BBOX_Regression_Pred = self.RPN_model.predict(input1, batch_size=1)
        print(RPN_Anchor_Pred.shape, RPN_BBOX_Regression_Pred.shape)

        # === Selection part ===
        # top_values, top_indices = tf.math.top_k()
        RPN_Anchor_Pred = tf.slice(RPN_Anchor_Pred, [0, 0, 0, 0, 1], [1, 23, 40, 9, 1])  # second channel is foreground
        print(RPN_Anchor_Pred.shape, RPN_BBOX_Regression_Pred.shape)
        RPN_Anchor_Pred = tf.squeeze(RPN_Anchor_Pred)
        RPN_BBOX_Regression_Pred = tf.squeeze(RPN_BBOX_Regression_Pred)
        shape1 = tf.shape(RPN_Anchor_Pred)
        print(RPN_Anchor_Pred.shape, RPN_BBOX_Regression_Pred.shape)
        RPN_Anchor_Pred = tf.reshape(RPN_Anchor_Pred, (-1,))
        n_anchor_proposal = 30
        top_values, top_indices = tf.math.top_k(RPN_Anchor_Pred, n_anchor_proposal)
        DebugPrint('top values', top_values)

        # test_indices = tf.where(tf.greater(tf.reshape(RPN_Anchor_Pred, (-1,)), 0.9))
        # print(test_indices)

        top_indices = tf.reshape(top_indices, (-1, 1))

        DebugPrint('top indices', top_indices)
        RPN_BBOX_Regression_Pred_shape = tf.shape(RPN_BBOX_Regression_Pred)
        RPN_BBOX_Regression_Pred = tf.reshape(RPN_BBOX_Regression_Pred, (-1, RPN_BBOX_Regression_Pred_shape[-1]))
        DebugPrint('RPN_BBOX_Regression_Pred shape', RPN_BBOX_Regression_Pred.shape)
        final_box_reg = tf.gather_nd(RPN_BBOX_Regression_Pred, top_indices)
        DebugPrint('final box reg values', final_box_reg)

        # Need to delete these two lines
        final_box_reg = np.array(final_box_reg)
        final_box_reg = final_box_reg / np.max(np.abs(final_box_reg))

        DebugPrint('final box reg shape', final_box_reg.shape)

        update_value = [2] * n_anchor_proposal
        RPN_Anchor_Pred = tf.tensor_scatter_nd_update(RPN_Anchor_Pred, top_indices, update_value)
        RPN_Anchor_Pred = tf.reshape(RPN_Anchor_Pred, shape1)
        Anchor_Pred_top_indices = tf.where(tf.equal(RPN_Anchor_Pred, 2))
        DebugPrint('original_indices shape', Anchor_Pred_top_indices.shape)
        DebugPrint('original_indices', Anchor_Pred_top_indices)
        base_boxes = tf.gather_nd(anchor_candidates, Anchor_Pred_top_indices)
        DebugPrint('base_boxes shape', base_boxes.shape)
        DebugPrint('base_boxes', base_boxes)
        base_boxes = np.array(base_boxes)

        final_box = bbox_tools.bbox_reg2truebox(base_boxes=base_boxes, regs=final_box_reg)


if __name__=='__main__':
    t1 = RoI(IMG_SHAPE=(720,1280,3))

