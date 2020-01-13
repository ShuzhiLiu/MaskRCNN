from Debugger import DebugPrint
import numpy as np
import tensorflow as tf
from NN_Parts import Backbone

class RoI:
    def __init__(self, backbone_model, IMG_SHAPE, n_stage= 5):
        self.base_model = backbone_model
        proposal_boxes = tf.keras.Input(shape=(4,),batch_size = None,name='PROPOSAL_BOXES')
        shape1 = tf.shape(proposal_boxes)
        n_boxes = tf.gather_nd(shape1, [0])
        indices = tf.ones(shape=n_boxes, dtype=tf.int32)
        img_shape_constant = tf.constant([IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[0], IMG_SHAPE[1]], tf.float32)
        proposal_boxes2 = tf.math.divide(proposal_boxes, img_shape_constant)



        image_crop = tf.image.crop_and_resize(self.base_model.output, proposal_boxes2,indices, [6,6])
        flatten1 = tf.keras.layers.GlobalAveragePooling2D()(image_crop)
        fc1 = tf.keras.layers.Dense(units=2048, activation='relu')(flatten1)
        fc2 = tf.keras.layers.Dense(units=2048, activation='relu')(fc1)
        class_header = tf.keras.layers.Dense(units=80, activation='softmax')(fc2)
        box_reg_header = tf.keras.layers.Dense(units=4, activation='linear')(fc2)

        self.RoI_model = tf.keras.Model(inputs=[self.base_model.input, proposal_boxes], outputs=[class_header, box_reg_header])
        tf.keras.utils.plot_model(self.RoI_model, 'RoI_with_backbone.png', show_shapes=True)



if __name__=='__main__':
    b1 = Backbone()
    t1 = RoI(b1.backbone_model,IMG_SHAPE=(720,1280,3))

