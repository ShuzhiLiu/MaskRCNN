from NN_Parts import Backbone, RPN, RoI
from Data_Helper import coco_tools
import numpy as np
from NN_Helper import NN_data_generator, gen_candidate_anchors, bbox_tools, RoI_generator
import tensorflow as tf
from Debugger import DebugPrint
from FasterRCNN_config import Param
import pickle
import os


class FasterRCNN():
    def __init__(self, IMG_SHAPE=Param.IMG_SHAPE):
        b1 = Backbone(IMG_SHAPE=IMG_SHAPE)
        self.IMG_SHAPE = IMG_SHAPE
        self.backbone_model = b1.backbone_model
        # === RPN part ===
        self.RPN = RPN(self.backbone_model, Param.LAMBDA_FACTOR, Param.BATCH)
        self.RPN_model = self.RPN.RPN_model
        self.RPN_train_model = self.RPN.RPN_train_model

        # === RoI part ===
        self.RoI = RoI(self.backbone_model, self.IMG_SHAPE, Param.N_STAGE)
        self.RoI_train_model = self.RoI.RoI_train_model

        # === Data part ===
        self.train_data_generator = NN_data_generator(
            file=f"{Param.PATH_DATA}/{Param.DATASET_ID}/annotations/train.json",
            imagefolder_path=Param.PATH_IMAGES)
        self.cocotool = self.train_data_generator.dataset_coco

        self.anchor_candidate_generator = gen_candidate_anchors(img_shape=(IMG_SHAPE[0], IMG_SHAPE[1]), n_stage=Param.N_STAGE)
        self.anchor_candidates = self.anchor_candidate_generator.anchor_candidates

    def test_loss_function(self):
        inputs, anchor_targets, bbox_targets = self.train_data_generator.gen_train_data_RPN()
        print(inputs.shape, anchor_targets.shape, bbox_targets.shape)
        input1 = np.reshape(inputs[0, :, :, :], (1, 720, 1280, 3))
        anchor1 = np.reshape(anchor_targets[0, :, :, :], (1, 23, 40, 9))
        anchor2 = tf.convert_to_tensor(anchor1)
        anchor2 = tf.dtypes.cast(anchor2, tf.int32)
        anchor2 = tf.one_hot(anchor2, 2, axis=-1)
        print(anchor1)
        bbox1 = np.reshape(bbox_targets[0, :, :, :, :], (1, 23, 40, 9, 4))
        loss = self.RPN._RPN_loss(anchor1, bbox1, anchor2, bbox1)
        print(loss)

    def test_proposal_visualization(self):
        # === Prediction part ===
        if os.path.isfile('train_data_RPN_temp.pkl'):
            with open('train_data_RPN_temp.pkl', 'rb') as f:
                inputs, anchor_targets, bbox_reg_targets = pickle.load(f)
        else:
            inputs, anchor_targets, bbox_reg_targets = self.train_data_generator.gen_train_data_RPN()
            with open('train_data_RPN_temp.pkl', 'wb') as f:
                pickle.dump([inputs, anchor_targets, bbox_reg_targets], f)
        print(inputs.shape, anchor_targets.shape, bbox_reg_targets.shape)
        input1 = np.reshape(inputs[0, :, :, :], (1, 720, 1280, 3))
        RPN_Anchor_Pred, RPN_BBOX_Regression_Pred = self.RPN_model.predict(input1, batch_size=1)
        print(RPN_Anchor_Pred.shape, RPN_BBOX_Regression_Pred.shape)

        # === Selection part ===
        # top_values, top_indices = tf.math.top_k()
        RPN_Anchor_Pred = tf.slice(RPN_Anchor_Pred, [0, 0, 0, 0, 1], [1, 23, 40, 9, 1])  # second channel is foreground
        print(RPN_Anchor_Pred.shape, RPN_BBOX_Regression_Pred.shape)
        # squeeze the pred of anchor and bbox_reg
        RPN_Anchor_Pred = tf.squeeze(RPN_Anchor_Pred)
        RPN_BBOX_Regression_Pred = tf.squeeze(RPN_BBOX_Regression_Pred)
        shape1 = tf.shape(RPN_Anchor_Pred)
        print(RPN_Anchor_Pred.shape, RPN_BBOX_Regression_Pred.shape)
        # flatten the pred of anchor to get top N values and indices
        RPN_Anchor_Pred = tf.reshape(RPN_Anchor_Pred, (-1,))
        n_anchor_proposal = Param.ANCHOR_PROPOSAL_N
        top_values, top_indices = tf.math.top_k(RPN_Anchor_Pred,
                                                n_anchor_proposal)  # top_k has sort function. it's important here
        top_indices = tf.gather_nd(top_indices, tf.where(tf.greater(top_values, Param.ANCHOR_THRESHOLD)))
        top_values = tf.gather_nd(top_values, tf.where(tf.greater(top_values, Param.ANCHOR_THRESHOLD)))

        DebugPrint('top values', top_values)

        # test_indices = tf.where(tf.greater(tf.reshape(RPN_Anchor_Pred, (-1,)), 0.9))
        # print(test_indices)

        top_indices = tf.reshape(top_indices, (-1, 1))
        DebugPrint('top indices', top_indices)
        update_value = tf.math.add(top_values, 1)
        RPN_Anchor_Pred = tf.tensor_scatter_nd_update(RPN_Anchor_Pred, top_indices, update_value)
        RPN_Anchor_Pred = tf.reshape(RPN_Anchor_Pred, shape1)

        # --- find the base boxes ---
        Anchor_Pred_top_indices = tf.where(tf.greater(RPN_Anchor_Pred, 1))
        DebugPrint('original_indices shape', Anchor_Pred_top_indices.shape)
        DebugPrint('original_indices', Anchor_Pred_top_indices)
        base_boxes = tf.gather_nd(self.anchor_candidates, Anchor_Pred_top_indices)
        DebugPrint('base_boxes shape', base_boxes.shape)
        DebugPrint('base_boxes', base_boxes)
        base_boxes = np.array(base_boxes)

        # --- find the bbox_regs ---
        # flatten the bbox_reg by last dim to use top_indices to get final_box_reg
        RPN_BBOX_Regression_Pred_shape = tf.shape(RPN_BBOX_Regression_Pred)
        RPN_BBOX_Regression_Pred = tf.reshape(RPN_BBOX_Regression_Pred, (-1, RPN_BBOX_Regression_Pred_shape[-1]))
        DebugPrint('RPN_BBOX_Regression_Pred shape', RPN_BBOX_Regression_Pred.shape)
        final_box_reg = tf.gather_nd(RPN_BBOX_Regression_Pred, top_indices)
        DebugPrint('final box reg values', final_box_reg)

        # Convert to numpy to plot
        final_box_reg = np.array(final_box_reg)
        DebugPrint('final box reg shape', final_box_reg.shape)
        final_box = bbox_tools.bbox_reg2truebox(base_boxes=base_boxes, regs=final_box_reg)

        # === Non maximum suppression ===
        final_box_temp = np.array(final_box).astype(np.int)
        nms_boxes_list = []
        while final_box_temp.shape[0] > 0:
            ious = self.nms_loop_np(final_box_temp)
            nms_boxes_list.append(
                final_box_temp[0, :])  # since it's sorted by the value, here we can pick the first one each time.
            final_box_temp = final_box_temp[ious < Param.RPN_NMS_THRESHOLD]
        print(len(nms_boxes_list))

        # Need to convert above instructions to tf operations

        # === visualization part ===
        # clip the boxes to make sure they are legal boxes
        final_box = bbox_tools.clip_boxes(final_box, self.IMG_SHAPE)

        original_boxes = self.cocotool.GetOriginalBboxesList(image_id=self.cocotool.image_ids[0])
        self.cocotool.DrawBboxes(Original_Image=input1[0], Bboxes=original_boxes, show=True)
        target_anchor_boxes, target_classes = self.train_data_generator.gen_target_anchor_bboxes_classes(image_id=self.cocotool.image_ids[0])
        self.cocotool.DrawBboxes(Original_Image=input1[0], Bboxes=target_anchor_boxes, show=True)
        self.cocotool.DrawBboxes(Original_Image=input1[0], Bboxes=base_boxes.tolist(), show=True)
        self.cocotool.DrawBboxes(Original_Image=input1[0], Bboxes=final_box.tolist(), show=True)
        self.cocotool.DrawBboxes(Original_Image=input1[0], Bboxes=nms_boxes_list, show=True)

    def test_RoI_visualization(self):
        # === prediction part ===
        input_images, target_anchor_bboxes, target_classes = self.train_data_generator._gen_train_data_RoI_one(self.train_data_generator.dataset_coco.image_ids[0])
        input_images, target_anchor_bboxes, target_classes = np.asarray(input_images).astype(np.float), np.asarray(target_anchor_bboxes), np.asarray(target_classes)
        # TODO:check tf.image.crop_and_resize
        input_images2 = input_images[:1].astype(np.float)
        print(input_images2.shape)
        target_anchor_bboxes2 = target_anchor_bboxes[:1].astype(np.float)
        print(target_anchor_bboxes2.shape)
        class_header, box_reg_header = self.RoI_train_model.predict([input_images2, target_anchor_bboxes2])
        print(class_header.shape,box_reg_header.shape)
        print(class_header)


    def nms_loop_np(self, boxes):
        # boxes : (N, 4), box_1target : (4,)
        # box axis format: (x1,y1,x2,y2)
        # boxes = boxes.astype(np.float)
        box_1target = np.ones(shape=boxes.shape)
        zeros = np.zeros(shape=boxes.shape)
        box_1target = box_1target * boxes[0, :]
        boxBArea = (box_1target[:, 2] - box_1target[:, 0] + 1) * (box_1target[:, 3] - box_1target[:, 1] + 1)
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.max(np.array([boxes[:, 0], box_1target[:, 0]]), axis=0)
        yA = np.max(np.array([boxes[:, 1], box_1target[:, 1]]), axis=0)
        xB = np.min(np.array([boxes[:, 2], box_1target[:, 2]]), axis=0)
        yB = np.min(np.array([boxes[:, 3], box_1target[:, 3]]), axis=0)

        # compute the area of intersection rectangle
        interArea = np.max(np.array([zeros[:, 0], xB - xA + 1]), axis=0) * np.max(np.array([zeros[:, 0], yB - yA + 1]),
                                                                                  axis=0)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        ious = (interArea / (boxAArea + boxBArea - interArea))

        # return the intersection over union value
        return ious

    def nms_loop_tf(self,boxes):
        ious = tf.numpy_function(func=self.nms_loop_np, inp=[boxes], Tout=tf.float32)

    def train_RPN(self, load_data=False):
        if load_data and os.path.isfile('train_data_RPN_temp.pkl'):
            with open('train_data_RPN_temp.pkl', 'rb') as f:
                inputs, anchor_targets, bbox_reg_targets = pickle.load(f)
        else:
            inputs, anchor_targets, bbox_reg_targets = self.train_data_generator.gen_train_data_RPN()
            with open('train_data_RPN_temp.pkl', 'wb') as f:
                pickle.dump([inputs, anchor_targets, bbox_reg_targets], f)
        self.RPN_train_model.fit([inputs, anchor_targets, bbox_reg_targets],
                                 batch_size=Param.BATCH,
                                 epochs=Param.EPOCH)
    def train_RoI(self):
        for _ in range(100):
            x,y = self.train_data_generator.gen_train_data_RoI_generator()
            self.RoI_train_model.fit(x=x,
                                     y=y,
                                     batch_size=4,
                                     epochs=1)
        # TODO: complete the generator below
        # generator = RoI_generator(self.train_data_generator)
        # self.RoI_train_model.fit_generator(generator, steps_per_epoch=25, epochs=1,max_queue_size=0,use_multiprocessing=False)

    def save_weight(self):
        self.RPN_model.save_weights(filepath=f"{Param.PATH_MODEL}/RPN_model")

    def load_weight(self):
        self.RPN_model.load_weights(filepath=f"{Param.PATH_MODEL}/RPN_model")


if __name__ == '__main__':
    f1 = FasterRCNN()
    # data1 = coco_tools(
    #     file='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data/1940091026744/annotations/train.json',
    #     imagefolder_path='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images')
    # img1 = data1.GetOriginalImage(image_id='20191119T063709-cca043ed-32fe-4da0-ba75-e4a12b88eef4')
    # t1, t2 = f1.RPN_model.predict(np.array([img1]))
    # print(t1, t2)
    # f1.test_proposal_visualization()
    # f1.train_RPN(load_data=True)
    # f1.save_weight()
    f1.load_weight()
    f1.test_proposal_visualization()
    # f1.test_RoI_visualization()
    # f1.train_RoI()
    # f1.test_RoI_visualization()
