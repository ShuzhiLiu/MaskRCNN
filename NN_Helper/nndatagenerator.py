import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from Data_Helper import CocoTools
from NN_Helper import BboxTools, GenCandidateAnchors


class NnDataGenerator():
    def __init__(self,
                 file: str,
                 imagefolder_path: str,
                 anchor_base_size: int,
                 ratios: list,
                 scales,
                 n_anchors: int,
                 img_shape_resize: tuple = (800, 1333, 3),
                 n_stage: int = 5,
                 threshold_iou_rpn: float = 0.7,
                 threshold_iou_roi: float = 0.55
                 ):
        # TODO: complete the resize part of the data generator
        self.threshold_iou_rpn = threshold_iou_rpn
        self.threshold_iou_roi = threshold_iou_roi
        self.dataset_coco = CocoTools(file, imagefolder_path, img_shape_resize)
        self.gen_candidate_anchors = GenCandidateAnchors(base_size=anchor_base_size, ratios=ratios, scales=scales,
                                                         img_shape=img_shape_resize, n_stage=n_stage,
                                                         n_anchors=n_anchors)
        self.img_shape_resize = img_shape_resize

    def _resize_img(self, img):
        return cv.resize(img, self.img_shape_resize, cv.INTER_LINEAR)

    def _resize_box(self, box):
        pass

    def gen_train_input_one(self, image_id):
        return self.dataset_coco.get_original_image(image_id=image_id)

    def gen_train_target_anchor_boxreg_for_rpn(self, image_id, debuginfo=False):
        bboxes = self.dataset_coco.get_original_bboxes_list(image_id=image_id)

        # === resize ===

        bboxes_ious = []  # for each gt_bbox calculate ious with candidates
        for bbox in bboxes:
            ious = BboxTools.ious(self.gen_candidate_anchors.anchor_candidates_list, bbox)
            ious_temp = np.ones(shape=(len(ious)), dtype=np.float) * 0.5
            # other author's implementations are use -1 to indicate ignoring, here use 0.5 to use max
            ious_temp = np.where(np.asarray(ious) > self.threshold_iou_rpn, 1, ious_temp)
            ious_temp = np.where(np.asarray(ious) < 0.3, 0, ious_temp)
            ious_temp[np.argmax(ious)] = 1
            bboxes_ious.append(ious_temp)

        # for each candidate anchor, determine the anchor target
        anchors_target = np.array(bboxes_ious)
        anchors_target = np.max(anchors_target, axis=0)
        anchors_target = np.reshape(anchors_target, newshape=(
            self.gen_candidate_anchors.h, self.gen_candidate_anchors.w, self.gen_candidate_anchors.n_anchors))
        if debuginfo:
            print(f"[Debug INFO] Number of total gt bboxes :{len(bboxes)}")
            print(
                f"[Debug INFO] Number of total target anchors: {anchors_target[np.where(anchors_target == 1)].shape[0]}")
            print(f"[Debug INFO] Shape of anchors_target: {anchors_target.shape}")
            print(
                f"[Debug INFO] Selected anchors: \n {self.gen_candidate_anchors.anchor_candidates[np.where(anchors_target == 1)]}")
        # test
        # self.anchor_generator.anchors_candidate[np.where(anchors_target==1)] = self.anchor_generator.anchors_candidate[np.where(anchors_target==1)] +100
        # print(f"Selected anchors: \n {self.anchor_generator.anchors_candidate[np.where(anchors_target == 1)]}")

        # for each gt_box, determine the box reg target
        bbox_reg_target = np.zeros(
            shape=(self.gen_candidate_anchors.h, self.gen_candidate_anchors.w, self.gen_candidate_anchors.n_anchors, 4),
            dtype=np.float)
        for index, bbox_ious in enumerate(bboxes_ious):
            ious_temp = np.reshape(bbox_ious, newshape=(
                self.gen_candidate_anchors.h, self.gen_candidate_anchors.w, self.gen_candidate_anchors.n_anchors))
            gt_box = bboxes[index]
            candidate_boxes = self.gen_candidate_anchors.anchor_candidates[np.where(ious_temp == 1)]
            # print(candidate_boxes,gt_box)
            box_reg = BboxTools.bbox_regression_target(candidate_boxes, gt_box)
            # print(box_reg)
            # print(bbox_tools.bbox_reg2truebox(candidate_boxes, box_reg))
            bbox_reg_target[np.where(ious_temp == 1)] = box_reg

        return anchors_target, bbox_reg_target

    def gen_target_anchor_bboxes_classes_for_debug(self, image_id, debuginfo=False):
        bboxes = self.dataset_coco.get_original_bboxes_list(image_id=image_id)
        sparse_targets = self.dataset_coco.get_original_category_sparse_list(image_id=image_id)

        bboxes_ious = []  # for each gt_bbox calculate ious with candidates
        for bbox in bboxes:
            ious = BboxTools.ious(self.gen_candidate_anchors.anchor_candidates_list, bbox)
            ious_temp = np.ones(shape=(len(ious)), dtype=np.float) * 0.5
            # other author's implementations are use -1 to indicate ignoring, here use 0.5 to use max
            ious_temp = np.where(np.asarray(ious) > self.threshold_iou_rpn, 1, ious_temp)
            ious_temp = np.where(np.asarray(ious) < 0.3, 0, ious_temp)
            ious_temp[np.argmax(ious)] = 1
            bboxes_ious.append(ious_temp)

        # for each gt_box, determine the box reg target
        target_anchor_bboxes = []
        target_classes = []
        for index, bbox_ious in enumerate(bboxes_ious):
            ious_temp = np.reshape(bbox_ious, newshape=(
                self.gen_candidate_anchors.h, self.gen_candidate_anchors.w, self.gen_candidate_anchors.n_anchors))
            candidate_boxes = self.gen_candidate_anchors.anchor_candidates[np.where(ious_temp == 1)]
            n = candidate_boxes.shape[0]
            for i in range(n):
                target_anchor_bboxes.append(candidate_boxes[i])
                target_classes.append(sparse_targets[index])
        return target_anchor_bboxes, target_classes

    def gen_train_data_rpn_one(self, image_id):
        input1 = self.gen_train_input_one(image_id)
        anchor_target, bbox_reg_target = self.gen_train_target_anchor_boxreg_for_rpn(image_id)
        return np.array([input1]).astype(np.float), np.array([anchor_target]).astype(np.float), np.array(
            [bbox_reg_target]).astype(np.float)

    def gen_train_data_rpn_all(self):
        inputs = []
        anchor_targets = []
        bbox_reg_targets = []
        for image_id in self.dataset_coco.image_ids:
            inputs.append(self.gen_train_input_one(image_id))
            anchor_target, bbox_reg_target = self.gen_train_target_anchor_boxreg_for_rpn(image_id)
            anchor_targets.append(anchor_target)
            bbox_reg_targets.append(bbox_reg_target)
        return np.array(inputs).astype(np.float), np.array(anchor_targets), np.array(bbox_reg_targets)

    def gen_train_data_roi_one(self, image_id, bbox_list):
        gt_bboxes = self.dataset_coco.get_original_bboxes_list(image_id=image_id)
        sparse_targets = self.dataset_coco.get_original_category_sparse_list(image_id=image_id)

        bboxes_ious = []  # for each gt_bbox calculate ious with candidates
        for bbox in gt_bboxes:
            ious = BboxTools.ious(bbox_list, bbox)
            ious_temp = np.zeros(shape=(len(ious)), dtype=np.float)
            # other author's implementations are use -1 to indicate ignoring, here use 0.5 to use max
            ious_temp = np.where(np.asarray(ious) > self.threshold_iou_roi, 1, ious_temp)
            ious_temp[np.argmax(ious)] = 1
            bboxes_ious.append(ious_temp)

        # for each gt_box, determine the box reg target
        original_img = self.gen_train_input_one(image_id)
        input_images = []
        input_box_filtered_by_iou = []
        target_classes = []
        target_bbox_reg = []
        for index_gt, bbox_ious in enumerate(bboxes_ious):
            candidate_boxes = np.asarray(bbox_list)[np.where(bbox_ious == 1)]
            n = candidate_boxes.shape[0]
            for i in range(n):
                input_box_filtered_by_iou.append(candidate_boxes[i].astype(np.float))
                box_reg = BboxTools.bbox_regression_target(pred_boxes=candidate_boxes[i].reshape((1, 4)),
                                                           gt_box=gt_bboxes[index_gt])
                target_bbox_reg.append(box_reg.ravel())
                target_classes.append(sparse_targets[index_gt])
                input_images.append(original_img.astype(np.float))
        for index_gt, bbox_gt in enumerate(gt_bboxes):
            input_images.append(original_img.astype(np.float))
            input_box_filtered_by_iou.append(bbox_gt.astype(np.float))
            target_classes.append(sparse_targets[index_gt])
            target_bbox_reg.append(np.array([0, 0, 0, 0], dtype=np.float))
        return np.asarray(input_images).astype(
            np.float), np.asarray(input_box_filtered_by_iou), np.asarray(target_classes), np.asarray(
            target_bbox_reg)

    def _validate_bbox(self, image_id, bboxes):
        img1 = self.dataset_coco.get_original_image(image_id=image_id)
        for bbox in bboxes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img1 = cv.rectangle(img1, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 4)
        plt.imshow(img1)
        plt.show()

    def _validata_masks(self, image_id):
        img1 = self.dataset_coco.get_original_image(image_id=image_id)
        temp_img = np.zeros(shape=img1.shape, dtype=np.uint8)
        masks = self.dataset_coco.get_original_segms_mask_list(image_id=image_id)
        for mask in masks:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            temp_img[:, :, 0][mask.astype(bool)] = color[0]
            temp_img[:, :, 1][mask.astype(bool)] = color[1]
            temp_img[:, :, 2][mask.astype(bool)] = color[2]
        img1 = (img1 * 0.5 + temp_img * 0.5).astype(np.uint8)
        plt.imshow(img1, cmap='gray')
        plt.show()


def test2():
    base_path = ''
    imagefolder_path = ''
    dataset_id = ''
    image_id = ''
    t1 = NnDataGenerator(file=f"{base_path}/{dataset_id}/annotations/train.json",
                         imagefolder_path=imagefolder_path)
    bboxes = t1.dataset_coco.get_original_bboxes_list(image_id=image_id)
    t1._validate_bbox(image_id=image_id, bboxes=bboxes)
    t1._validata_masks(image_id=image_id)
    t1.gen_train_target_anchor_boxreg_for_rpn(image_id=image_id)


def test():
    base_path = ''
    imagefolder_path = ''
    dataset_id = ''
    image_id = ''
    data1 = CocoTools(json_file=f"{base_path}/{dataset_id}/annotations/train.json",
                      image_folder_path=imagefolder_path)
    img1 = data1.get_original_image(image_id=image_id)
    print(data1.images)
    bboxes = data1.get_original_bboxes_list(image_id=image_id)
    print(bboxes)
    # img1 = np.zeros(shape=(720,1280,3), dtype=np.uint8)
    for bbox in bboxes:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # cv.rectangle(img1,(bbox[0],bbox[1]), (bbox[2],bbox[3]), 255)
        # print(bbox)
        # cv.rectangle(img=img1,rec=(bbox[1],bbox[0],bbox[3]-bbox[1],bbox[2]-bbox[0]), color=color, thickness=4)
        cv.rectangle(img1, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 4)
    plt.imshow(img1)
    plt.show()

    g1 = GenCandidateAnchors()
    print(len(g1.anchor_candidates_list))
    ious = BboxTools.ious(g1.anchor_candidates_list, bboxes[0])
    ious[np.argmax(ious)] = 1
    print(len(ious))
    ious_np = np.reshape(ious, newshape=(23, 40, 9))
    index = np.where(ious_np == 1)
    print(index)
