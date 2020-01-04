from NN_Parts import Backbone, RPN
from Data_Helper import coco_tools
import numpy as np
from NN_Helper import gen_train_target
import tensorflow as tf


class FasterRCNN():
    def __init__(self):
        b1 = Backbone()
        self.backbone_model = b1.backbone_model
        self.RPN = RPN(self.backbone_model)
        self.RPN_model = self.RPN.RPN_model
        self.RPN_train_model = self.RPN.RPN_train_model

        BASE_PATH = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data'
        imagefolder_path = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images'
        DATASET_ID = '1940091026744'
        image_id = '20191119T063709-cca043ed-32fe-4da0-ba75-e4a12b88eef4'
        self.train_data_generator = gen_train_target(file=f"{BASE_PATH}/{DATASET_ID}/annotations/train.json",
                              imagefolder_path=imagefolder_path)


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


    def train(self):
        inputs, anchor_targets, bbox_targets = self.train_data_generator.gen_train_data()
        self.RPN_train_model.fit([inputs, anchor_targets, bbox_targets],
                                 batch_size=1,
                                 epochs=12)




if __name__=='__main__':
    f1 = FasterRCNN()
    # data1 = coco_tools(
    #     file='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data/1940091026744/annotations/train.json',
    #     imagefolder_path='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images')
    # img1 = data1.GetOriginalImage(image_id='20191119T063709-cca043ed-32fe-4da0-ba75-e4a12b88eef4')
    # t1, t2 = f1.RPN_model.predict(np.array([img1]))
    # print(t1, t2)
    f1.train()
        