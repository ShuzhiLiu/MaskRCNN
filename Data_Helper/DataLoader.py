import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Data_Helper import coco_tools
import random
from NN_Helper import bbox_tools, gen_candidate_anchors

BASE_PATH = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data'
DATASET_ID = '1945415016934'

with open(f"{BASE_PATH}/{DATASET_ID}/annotations/train.json", 'r') as f:
    train_coco = json.load(f)

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
ious = bbox_tools.ious(g1.anchor_candidates_list, bboxes[0])
print(ious)