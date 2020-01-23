import numpy as np
import tensorflow as tf
from NN_Helper import bbox_tools

class bbox_tools_tf:
    # Beside ious, format of boxes is numpy.
    # format of boxes is list for ious
    # TODO: deal with ious function
    @classmethod
    def ious(cls, boxes_list, box_1target):
        # box axis format: (x1,y1,x2,y2)
        boxBArea = (box_1target[2] - box_1target[0] + 1) * (box_1target[3] - box_1target[1] + 1)
        ious = []
        for box in boxes_list:
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(box[0], box_1target[0])
            yA = max(box[1], box_1target[1])
            xB = min(box[2], box_1target[2])
            yB = min(box[3], box_1target[3])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            ious.append(interArea / float(boxAArea + boxBArea - interArea))

        # return the intersection over union value
        return ious

    @classmethod
    def bbox_regression_target(cls, pred_boxes, gt_box):
        reg_target = tf.numpy_function(bbox_tools.bbox_regression_target, [pred_boxes, gt_box], tf.float32)
        return reg_target

    @classmethod
    def bbox_reg2truebox(cls, base_boxes, regs):
        # input shape (N,4) , (N,4)
        # tested
        truebox = tf.numpy_function(bbox_tools.bbox_reg2truebox, [base_boxes, regs], tf.int32)
        return  truebox

    @classmethod
    def xxyy2xywh(cls, boxes):
        xywh = tf.numpy_function(bbox_tools.xxyy2xywh, [boxes], tf.float32)
        return xywh

    @classmethod
    def xywh2xxyy(cls, boxes):
        xyxy = tf.numpy_function(bbox_tools.xywh2xxyy, [boxes], tf.int32)
        return xyxy

    @classmethod
    def clip_boxes(cls, boxes, IMG_SHAPE):
        boxes2 = tf.numpy_function(bbox_tools.clip_boxes, [boxes, IMG_SHAPE], tf.int32)
        return boxes2


if __name__=='__main__':
    t1 = tf.constant([[10,10,20,20]], dtype=tf.int32)
    t2 = tf.constant([[5,5, 35, 35]])
    print(t1)
    print(bbox_tools_tf.xxyy2xywh(t1))
    print(bbox_tools_tf.xywh2xxyy(bbox_tools_tf.xxyy2xywh(t1)))

    print(bbox_tools_tf.xxyy2xywh(t2))
    print(bbox_tools_tf.bbox_regression_target(t1, t2))
    print(bbox_tools_tf.bbox_reg2truebox(t1, bbox_tools_tf.bbox_regression_target(t1, t2)))