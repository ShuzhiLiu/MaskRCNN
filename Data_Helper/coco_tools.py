import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
class coco_tools:
    def __init__(self, file, imagefolder_path):
        self.imagefolder_path = imagefolder_path
        with open(file, 'r') as f:
            dict1 = json.load(f)
        self.info = dict1["info"]
        self.licenses = dict1["licenses"]
        self.images = dict1["images"]
        self.annotations = dict1["annotations"]
        self.categories = dict1["categories"]
        self.segment_info = dict1["segment_info"]
        self.image_ids = []
        for image in self.images:
            if image['id'] not in self.image_ids:
                self.image_ids.append(image['id'])



    def DrawSegmFromAnnoCoco(self, image_id, Original_Image, annos, show=False, savefile=False):
        height, width = 0, 0
        bboxes = []
        for anno in annos:
            if anno['image_id'] == image_id:
                height = anno['height']
                width = anno['width']
                bboxes.append(anno['bbox'])
        bboxes_int = np.array(bboxes, dtype=np.int)
        tempimg = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        masks_pred, class_names = self.GetSegmMaskFromAnnoCOCO(annos, image_id)
        _, _, n_masks = masks_pred.shape
        for i in range(n_masks):
            color_random = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_random2 = color_random.tolist()
            cv2.rectangle(tempimg,
                          (bboxes_int[i, 0], bboxes_int[i, 1]),
                          (bboxes_int[i, 0] + bboxes_int[i, 2], bboxes_int[i, 1] + bboxes_int[i, 3]),
                          color_random2[0],
                          2)
            cv2.putText(tempimg, f'{i + 1}th Object_{class_names[i]}',
                        (bboxes_int[i, 0] + bboxes_int[i, 2] + 10, bboxes_int[i, 1] + bboxes_int[i, 3]),
                        0, 0.3, color_random2[0])
            tempimg[:, :, 0][masks_pred[:, :, i]] = color_random[0, 0]
            tempimg[:, :, 1][masks_pred[:, :, i]] = color_random[0, 1]
            tempimg[:, :, 2][masks_pred[:, :, i]] = color_random[0, 2]
        Original_Image = (Original_Image * 0.5 + tempimg * 0.5).astype(np.uint8)
        plt.imshow(Original_Image)
        if show:
            plt.show()
        if savefile:
            plt.savefig(f'{os.getcwd()}/Images_Drawn/{image_id}.jpg', dpi=300)

    def DrawBboxes(self, Original_Image, Bboxes, show=False, savefile=False):
        # bbox is numpy format (x1, y1, x2, y2)
        height, width = Original_Image.shape[0], Original_Image.shape[1]
        tempimg = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        for bbox in Bboxes:
            color_random = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_random2 = color_random.tolist()
            cv2.rectangle(tempimg,
                          (bbox[1], bbox[0]),
                          (bbox[3], bbox[2]),
                          color_random2[0],
                          2)
        Original_Image = (Original_Image * 0.5 + tempimg * 0.5).astype(np.uint8)
        plt.imshow(Original_Image)
        if show:
            plt.show()
        if savefile:
            plt.savefig(f'{os.getcwd()}/Images_Drawn/11111111111.jpg', dpi=300)

    def GetSegmMaskFromAnnoCOCO(self, annos, image_id):
        segms = []
        height = None
        width = None
        class_ids = []
        for anno in annos:
            if anno['image_id'] == image_id:
                segms.append(np.reshape(anno['segmentation'][0], newshape=(-1, 2)))
                height = anno['height']
                width = anno['width']
                class_ids.append(anno['category_id'])
        n_segms = len(segms)
        mask_temp = np.zeros(shape=(height, width, n_segms), dtype=np.uint8)
        for i in range(n_segms):
            temp_one_mask = np.zeros(shape=(height, width), dtype=np.uint8)
            contour = (np.array(segms[i])).astype(int)
            cv2.fillPoly(img=temp_one_mask, pts=[contour], color=1)
            mask_temp[:, :, i] = temp_one_mask
        return mask_temp.astype(np.bool), class_ids

    def LoadAnnoCOCO(self, file, imagefolder_path):
        self.imagefolder_path = imagefolder_path
        with open(file, 'r') as f:
            dict1 = json.load(f)
        self.info = dict1["info"]
        self.licenses = dict1["licenses"]
        self.images = dict1["images"]
        self.annotations = dict1["annotations"]
        self.categories = dict1["categories"]
        self.segment_info = dict1["segment_info"]
        self.image_ids = []
        for image in self.images:
            if image['id'] not in self.image_ids:
                self.image_ids.append(image['id'])

    def GetOriginalImage(self,image_id):
        image_name = self.GetImageName(image_id)
        img = cv2.imread(f"{self.imagefolder_path}/{image_name}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.array(img_rgb)


    def GetOriginalBboxesList(self, image_id):
        # read original opencv bbox and convert to numpy format bbox
        Bboxes = []
        for anno in self.annotations:
            if anno['image_id'] == image_id:
                bbox = np.array(anno['bbox'], dtype=np.int)
                bbox[0], bbox[1],bbox[2],bbox[3] = bbox[1],bbox[0], bbox[1] + bbox[3], bbox[0]+bbox[2]
                Bboxes.append(bbox)
        return Bboxes

    def GetOriginalSegmsMaskList(self, image_id):
        # TODO: put mask list to dictionary of labels
        width, height = 0, 0
        for image in self.images:
            if image['id'] == image_id:
                width = image['width']
                height = image['height']
                break
        Masks = []
        for anno in self.annotations:
            if anno['image_id'] == image_id:
                img_temp = np.zeros(shape=(height,width), dtype=np.uint8)
                contour = np.reshape(anno['segmentation'], newshape=(-1,2)).astype(int)
                img_temp = cv2.fillPoly(img_temp, [contour], 1)
                Masks.append(img_temp)
        return Masks





    def GetImageName(self, image_id):
        for image in self.images:
            if image['id'] == image_id:
                return image['file_name']

    def DrawWithImageID(self, image_id):
        original_image = self.GetOriginalImage(image_id)
        self.DrawSegmFromAnnoCoco(image_id, original_image, self.annotations, True)
