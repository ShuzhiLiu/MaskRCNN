import numpy as np


class bbox_tools:


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
        '''
        both or inputs are numpy arrays
        :param pred_boxes: expected box (batchsize, x1, y1, x2, y2)
        :param gt_boxes: ground truth box (batchsize, x1, y1, x2, y2)
        :return: transforms
        '''
        gt_boxes = np.zeros(shape=pred_boxes.shape) + gt_box
        reg_target = np.zeros(shape=pred_boxes.shape)
        ex_boxes_xywh = cls.xxyy2xywh(pred_boxes)
        gt_boxes_xywh = cls.xxyy2xywh(gt_boxes)

        # The purpose of these procedure is to make sure target label in [-1,1] !!!
        # This can be achieved only when the iou>0.7, in the case the biggest iou is still small
        # the value will be out of [-1,1]
        reg_target[:, 0] = (gt_boxes_xywh[:, 0] - ex_boxes_xywh[:, 0]) / ex_boxes_xywh[:, 2]
        reg_target[:, 1] = (gt_boxes_xywh[:, 1] - ex_boxes_xywh[:, 1]) / ex_boxes_xywh[:, 3]
        reg_target[:, 2] = np.log(gt_boxes_xywh[:, 2] / ex_boxes_xywh[:, 2])
        reg_target[:, 3] = np.log(gt_boxes_xywh[:, 3] / ex_boxes_xywh[:, 3])

        return reg_target

    @classmethod
    def bbox_reg2truebox(cls, base_box, reg):
        # input shape (N,4) , (N,4)
        # tested
        box_after_reg = np.zeros(shape=base_box.shape)
        base_box_xywh = cls.xxyy2xywh(base_box)
        box_after_reg[:,0] = reg[:,0] * base_box_xywh[:,2] + base_box_xywh[:,0]
        box_after_reg[:,1] = reg[:,1] * base_box_xywh[:,3] + base_box_xywh[:,1]
        box_after_reg[:,2] = np.exp(reg[:,2]) * base_box_xywh[:,2]
        box_after_reg[:,3] = np.exp(reg[:,3]) * base_box_xywh[:,3]

        box_after_reg = cls.xywh2xxyy(box_after_reg)
        return box_after_reg

    @classmethod
    def bbox_transform_inv(cls, bbox, deltas):
        bbox_xywh = cls.xxyy2xywh(bbox)

        pred_ctr_x = deltas[:, 0] * bbox_xywh[:, 2] + bbox_xywh[:, 0]
        pred_ctr_y = deltas[:, 1] * bbox_xywh[:, 3] + bbox_xywh[:, 1]
        pred_w = np.exp(deltas[:, 2]) * bbox_xywh[:, 2]
        pred_h = np.exp(deltas[:, 3]) * bbox_xywh[:, 3]

        pred_ctr_x = pred_ctr_x.view(-1, 1)
        pred_ctr_y = pred_ctr_y.view(-1, 1)
        pred_w = pred_w.view(-1, 1)
        pred_h = pred_h.view(-1, 1)

        pred_box = np.cat([pred_ctr_x, pred_ctr_y, pred_w, pred_h], dim=1)

        return cls.xywh2xxyy(pred_box)

    @classmethod
    def bbox_transform_inv_cls(cls, boxes, deltas):
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
        pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
        pred_w = np.exp(dw) * widths.unsqueeze(1)
        pred_h = np.exp(dh) * heights.unsqueeze(1)

        pred_boxes = deltas.clone()
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    @classmethod
    def clip_boxes(cls, boxes, im_info):
        boxes[:, 0::4].clamp_(0, im_info[1] - 1)
        boxes[:, 1::4].clamp_(0, im_info[0] - 1)
        boxes[:, 2::4].clamp_(0, im_info[1] - 1)
        boxes[:, 3::4].clamp_(0, im_info[0] - 1)

        return boxes

    @classmethod
    def clip_boxes_cls(cls, boxes, im_shape):
        boxes[:, 0::4].clamp_(0, im_shape[1] - 1)
        boxes[:, 1::4].clamp_(0, im_shape[0] - 1)
        boxes[:, 2::4].clamp_(0, im_shape[1] - 1)
        boxes[:, 3::4].clamp_(0, im_shape[0] - 1)

        return boxes

    @classmethod
    def xxyy2xywh(cls, boxes):
        xywh = np.zeros(shape=boxes.shape)
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0] + 1
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1] + 1
        xywh[:, 0] = boxes[:, 0] + xywh[:, 2] / 2
        xywh[:, 1] = boxes[:, 1] + xywh[:, 3] / 2

        return xywh.astype(np.int)

    @classmethod
    def xywh2xxyy(cls, boxes):
        xyxy = np.zeros(shape=boxes.shape)
        xyxy[:, 0] = boxes[:, 0] - (boxes[:, 2] - 1) / 2
        xyxy[:, 1] = boxes[:, 1] - (boxes[:, 3] - 1) / 2
        xyxy[:, 2] = boxes[:, 0] + (boxes[:, 2] - 1) / 2
        xyxy[:, 3] = boxes[:, 1] + (boxes[:, 3] - 1) / 2

        return xyxy.astype(np.int)


if __name__ == '__main__':
    # test the bbox_tools
    image_shape = (720, 1280)  # (h, w) numpy format
    bbox1_xyxy = np.array([[0, 0, 9, 9]])  # (x,y,x,y)
    bbox1_whc = np.array([[5, 5, 10, 10], [5, 5, 10, 10]])
    bbox2_xyxy = np.array([[5, 5, 14, 14]])
    bbox2_whc = np.array([[10, 10, 10, 10]])
    print(bbox_tools.xxyy2xywh(bbox1_xyxy))
    print(bbox_tools.xywh2xxyy(bbox1_whc))
    print(bbox_tools.ious([[0, 0, 9, 9], [0, 0, 9, 9]], [5, 5, 14, 14]))
    print(bbox_tools.bbox_regression_target(bbox1_xyxy, bbox2_xyxy))
