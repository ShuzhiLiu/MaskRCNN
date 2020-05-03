import numpy as np
import tensorflow as tf

from NN_Helper import BboxTools


class BboxToolsTf:
    # Beside ious, format of boxes is numpy.
    # format of boxes is list for ious
    # TODO: tf.numpy_function only works on eager mode, not graph mode!!
    @classmethod
    def _ious(cls, boxes_np, box_1target):
        # box axis format: (x1,y1,x2,y2)
        # boxes_np:(?,4), box_1target:(4,)
        shape = boxes_np.shape
        box_b_area = (box_1target[2] - box_1target[0] + 1) * (box_1target[3] - box_1target[1] + 1)
        ious = np.zeros(shape=shape[0])
        for i in range(shape[0]):
            # determine the (x, y)-coordinates of the intersection rectangle
            box = boxes_np[i]
            x_a = max(box[0], box_1target[0])
            y_a = max(box[1], box_1target[1])
            x_b = min(box[2], box_1target[2])
            y_b = min(box[3], box_1target[3])

            # compute the area of intersection rectangle
            inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            box_a_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            ious[i] = (inter_area / float(box_a_area + box_b_area - inter_area))

        # return the intersection over union value
        return ious

    @classmethod
    def ious(cls, boxes_np, box_1target):
        ious = tf.numpy_function(cls._ious, [boxes_np, box_1target], tf.float32)
        return ious

    @classmethod
    def bbox_regression_target(cls, pred_boxes, gt_box):
        reg_target = tf.numpy_function(BboxTools.bbox_regression_target, [pred_boxes, gt_box], tf.float32)
        return reg_target

    @classmethod
    def bbox_reg2truebox(cls, base_boxes, regs):
        # input shape (N,4) , (N,4)
        # tested
        truebox = tf.numpy_function(BboxTools.bbox_reg2truebox, [base_boxes, regs], tf.float32)
        return truebox

    @classmethod
    def xxyy2xywh(cls, boxes):
        xywh = tf.numpy_function(BboxTools.xxyy2xywh, [boxes], tf.float32)
        return xywh

    @classmethod
    def xywh2xxyy(cls, boxes):
        xyxy = tf.numpy_function(BboxTools.xywh2xxyy, [boxes], tf.int32)
        return xyxy

    @classmethod
    def clip_boxes(cls, boxes, img_shape):
        boxes2 = tf.numpy_function(BboxTools.clip_boxes, [boxes, img_shape], tf.int32)
        return boxes2


if __name__ == '__main__':
    pass
    # t1 = tf.constant([[10, 10, 20, 20]], dtype=tf.int32)
    # t2 = tf.constant([[5, 5, 35, 35]])
    # print(t1)
    # print(BboxToolsTf.xxyy2xywh(t1))
    # print(BboxToolsTf.xywh2xxyy(BboxToolsTf.xxyy2xywh(t1)))
    #
    # print(BboxToolsTf.xxyy2xywh(t2))
    # print(BboxToolsTf.bbox_regression_target(t1, t2))
    # print(BboxToolsTf.bbox_reg2truebox(t1, BboxToolsTf.bbox_regression_target(t1, t2)))
    # print(BboxToolsTf.ious(t1, t2[0]))
