import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Data_Helper import coco_tools
import random
from NN_Helper import bbox_tools, gen_candidate_anchors



class gen_train_target():
    def __init__(self, file, imagefolder_path, img_shape=(720,1280), n_stage=5):
        self.dataset = coco_tools(file, imagefolder_path)
        self.anchor_generator = gen_candidate_anchors(img_shape=img_shape, n_stage=n_stage)

    def gen_train_target(self, image_id):
        bboxes = t1.dataset.GetOriginalBboxesList(image_id=image_id)
        bboxes_ious = []    # for each gt_bbox calculate ious with candidates
        for bbox in bboxes:
            ious = bbox_tools.ious(self.anchor_generator.anchors_candidate_list, bbox)
            ious_temp = np.ones(shape=(len(ious)), dtype=np.float) * 0.5
            # other author's implementations are use -1 to indicate ignoring, here use 0.5 to use max
            ious_temp = np.where(np.asarray(ious)>0.7, 1, ious_temp)
            ious_temp = np.where(np.asarray(ious)<0.3, 0, ious_temp)
            ious_temp[np.argmax(ious)] = 1
            bboxes_ious.append(ious_temp)

        anchors_target = np.array(bboxes_ious)
        anchors_target = np.max(anchors_target, axis=0)
        anchors_target = np.reshape(anchors_target, newshape=(self.anchor_generator.h, self.anchor_generator.w, self.anchor_generator.n_anchors))
        print(anchors_target.shape)
        print(anchors_target[np.where(anchors_target==1)])
        bbox_reg_target = None
        bbox_inside_weight = None
        bbox_outside_weight = None


        return anchors_target, bbox_reg_target, bbox_inside_weight, bbox_outside_weight


    def _validate_bbox(self,image_id, bboxes):
        img1 = self.dataset.GetOriginalImage(image_id=image_id)
        for bbox in bboxes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img1 = cv.rectangle(img1, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 4)
        plt.imshow(img1)
        plt.show()

    def _validata_masks(self, image_id):
        img1 = self.dataset.GetOriginalImage(image_id=image_id)
        temp_img = np.zeros(shape=img1.shape, dtype=np.uint8)
        Masks = self.dataset.GetOriginalSegmsMaskList(image_id=image_id)
        for mask in Masks:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            temp_img[:, :, 0][mask.astype(bool)] = color[0]
            temp_img[:, :, 1][mask.astype(bool)] = color[1]
            temp_img[:, :, 2][mask.astype(bool)] = color[2]
        img1 = (img1 * 0.5 + temp_img * 0.5).astype(np.uint8)
        plt.imshow(img1, cmap='gray')
        plt.show()



BASE_PATH = '/Users/liushuzhi/Google Drive/KyoceraRobotAI/mmdetection_tools/data'
DATASET_ID = '1945415016934'

t1 = gen_train_target(file=f"{BASE_PATH}/{DATASET_ID}/annotations/train.json",
                   imagefolder_path='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images')
bboxes = t1.dataset.GetOriginalBboxesList(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474")
# t1._validate_bbox(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474", bboxes=bboxes)
# t1._validata_masks(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474")
t1.gen_train_target(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474")


def test():
    data1 = coco_tools(file=f"{BASE_PATH}/{DATASET_ID}/annotations/train.json",
                       imagefolder_path='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images')
    img1 = data1.GetOriginalImage(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474")
    print(data1.images)
    bboxes = data1.GetOriginalBboxesList(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474")
    print(bboxes)
    # img1 = np.zeros(shape=(720,1280,3), dtype=np.uint8)
    for bbox in bboxes:
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        # cv.rectangle(img1,(bbox[0],bbox[1]), (bbox[2],bbox[3]), 255)
        # print(bbox)
        # cv.rectangle(img=img1,rec=(bbox[1],bbox[0],bbox[3]-bbox[1],bbox[2]-bbox[0]), color=color, thickness=4)
        cv.rectangle(img1, (bbox[1],bbox[0]), (bbox[3],bbox[2]), color, 4)
    plt.imshow(img1)
    plt.show()

    g1 = gen_candidate_anchors()
    print(len(g1.anchors_candidate_list))
    ious = bbox_tools.ious(g1.anchors_candidate_list, bboxes[0])
    ious[np.argmax(ious)] = 1
    print(len(ious))
    ious_np = np.reshape(ious, newshape=(23,40,9))
    index = np.where(ious_np==1)
    print(index)


