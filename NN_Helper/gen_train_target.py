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
            ious[np.argmax(ious)] = 1
            ious = np.array(ious).reshape((self.anchor_generator.h, self.anchor_generator.w, self.anchor_generator.n_anchors))
            ious = np.where(ious>0.7, 1, ious)
            ious = np.where(0.3<ious<0.7, -1, ious)
            ious = np.where(ious<0.3, 0, ious)
            bboxes_ious.append(ious)
        train_anchors = None
        train_bbox_reg = None

        return train_anchors, train_bbox_reg


    def _validate_bbox(self,image_id, bboxes):
        img1 = self.dataset.GetOriginalImage(image_id=image_id)
        for bbox in bboxes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv.rectangle(img1, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 4)
        plt.imshow(img1)
        plt.show()



BASE_PATH = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data'
DATASET_ID = '1945415016934'

t1 = gen_train_target(file=f"{BASE_PATH}/{DATASET_ID}/annotations/train.json",
                   imagefolder_path='/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images')
bboxes = t1.dataset.GetOriginalBboxesList(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474")
t1._validate_bbox(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474", bboxes=bboxes)
Masks = t1.dataset.GetOriginalSegmsMaskList(image_id="20191119T063434-f7b72bed-b7ad-48c8-870a-7b4eaad23474")
for mask in Masks:
    plt.imshow(mask, cmap='gray')
    plt.show()


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


