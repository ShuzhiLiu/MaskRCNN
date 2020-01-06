from NN_Parts import Backbone, RPN
from Data_Helper import coco_tools
import numpy as np
from NN_Helper import NN_data_generator, gen_candidate_anchors, bbox_tools
import tensorflow as tf
from Debugger import DebugPrint


class FasterRCNN():
    def __init__(self,IMG_SHAPE=(720, 1280, 3)):
        b1 = Backbone(IMG_SHAPE=IMG_SHAPE)
        self.IMG_SHAPE = IMG_SHAPE
        self.backbone_model = b1.backbone_model
        self.RPN = RPN(self.backbone_model)
        self.RPN_model = self.RPN.RPN_model
        self.RPN_train_model = self.RPN.RPN_train_model

        # BASE_PATH = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data'
        # imagefolder_path = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images'
        BASE_PATH = '/home/liushuzhi/Documents/mmdetection_tools/data'
        imagefolder_path = '/home/liushuzhi/Documents/mmdetection_tools/LocalData_Images'
        DATASET_ID = '1940091026744'
        image_id = '20191119T063709-cca043ed-32fe-4da0-ba75-e4a12b88eef4'
        self.train_data_generator = NN_data_generator(file=f"{BASE_PATH}/{DATASET_ID}/annotations/train.json",
                                                      imagefolder_path=imagefolder_path)
        self.cocotool = self.train_data_generator.dataset_coco

        self.anchor_candidate_generator = gen_candidate_anchors(img_shape=(IMG_SHAPE[0],IMG_SHAPE[1]))
        self.anchor_candidates = self.anchor_candidate_generator.anchor_candidates


    def test_loss_function(self):
        inputs, anchor_targets, bbox_targets =self.train_data_generator.gen_train_data()
        print(inputs.shape, anchor_targets.shape, bbox_targets.shape)
        input1 = np.reshape(inputs[0,:,:,:], (1, 720, 1280, 3))
        anchor1 = np.reshape(anchor_targets[0,:,:,:], (1, 23, 40, 9))
        anchor2 = tf.convert_to_tensor(anchor1)
        anchor2 = tf.dtypes.cast(anchor2, tf.int32)
        anchor2 = tf.one_hot(anchor2, 2, axis=-1)
        print(anchor1)
        bbox1 = np.reshape(bbox_targets[0,:,:,:,:], (1, 23, 40, 9, 4))
        loss = self.RPN._RPN_loss(anchor1, bbox1, anchor2, bbox1)
        print(loss)

    def test_proposal_visualization(self):
        # === Prediction part ===
        inputs, anchor_targets, bbox_targets = self.train_data_generator.gen_train_data()
        print(inputs.shape, anchor_targets.shape, bbox_targets.shape)
        input1 = np.reshape(inputs[0, :, :, :], (1, 720, 1280, 3))
        RPN_Anchor_Pred, RPN_BBOX_Regression_Pred = self.RPN_model.predict(input1, batch_size=1)
        print(RPN_Anchor_Pred.shape, RPN_BBOX_Regression_Pred.shape)

        # === Selection part ===
        # top_values, top_indices = tf.math.top_k()
        RPN_Anchor_Pred = tf.slice(RPN_Anchor_Pred, [0,0,0,0,1], [1,23,40,9,1]) # second channel is foreground
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

        top_indices = tf.reshape(top_indices, (-1,1))



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
        base_boxes = tf.gather_nd(self.anchor_candidates, Anchor_Pred_top_indices)
        DebugPrint('base_boxes shape', base_boxes.shape)
        DebugPrint('base_boxes', base_boxes)
        base_boxes = np.array(base_boxes)


        final_box = bbox_tools.bbox_reg2truebox(base_boxes=base_boxes, regs=final_box_reg)

        # Need to convert above instructions to tf operations

        # === visualization part ===

        # clip the boxes to make sure they are legal boxes
        x_max, y_max = self.IMG_SHAPE[0], self.IMG_SHAPE[1]
        final_box[:, 0][final_box[:, 0] < 0] = 0
        final_box[:, 1][final_box[:, 1] < 0] = 0
        final_box[:, 2][final_box[:, 2] > x_max] = x_max
        final_box[:, 3][final_box[:, 3] > y_max] = y_max

        self.cocotool.DrawBboxes(Original_Image=input1[0],Bboxes=base_boxes.tolist(), show=True)
        true_boxes = self.train_data_generator.gen_true_bbox_candidates(image_id=self.cocotool.image_ids[0])
        self.cocotool.DrawBboxes(Original_Image=input1[0],Bboxes=true_boxes, show=True)
        original_boxes = self.cocotool.GetOriginalBboxesList(image_id=self.cocotool.image_ids[0])
        self.cocotool.DrawBboxes(Original_Image=input1[0], Bboxes=original_boxes, show=True)



    def train(self):
        inputs, anchor_targets, bbox_reg_targets = self.train_data_generator.gen_train_data()
        self.RPN_train_model.fit([inputs, anchor_targets, bbox_reg_targets],
                                 batch_size=1,
                                 epochs=24)




if __name__=='__main__':
    f1 = FasterRCNN()
    # data1 = coco_tools(
    #     file='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data/1940091026744/annotations/train.json',
    #     imagefolder_path='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images')
    # img1 = data1.GetOriginalImage(image_id='20191119T063709-cca043ed-32fe-4da0-ba75-e4a12b88eef4')
    # t1, t2 = f1.RPN_model.predict(np.array([img1]))
    # print(t1, t2)
    f1.test_proposal_visualization()
    f1.train()
    f1.test_proposal_visualization()