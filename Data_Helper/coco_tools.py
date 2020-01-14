import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pycococreatortools import pycococreatortools
from copy import deepcopy


class coco_tools:
    def __init__(self, file, imagefolder_path):
        self.LoadAnnoCOCO(file, imagefolder_path)

    def DrawSegmFromAnnoCoco(self, image_id, Original_Image, annos, show=False, savefile=False):
        height, width = self.GetImageShape(image_id)
        bboxes = []
        for anno in annos:
            if anno['image_id'] == image_id:
                bboxes.append(anno['bbox'])
        bboxes_int = np.array(bboxes, dtype=np.int)
        tempimg = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        masks_pred, class_ids = self.GetSegmMaskFromAnnoCOCO(annos, image_id)
        _, _, n_masks = masks_pred.shape
        for i in range(n_masks):
            color_random = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_random2 = color_random.tolist()
            cv2.rectangle(tempimg,
                          (bboxes_int[i, 0], bboxes_int[i, 1]),
                          (bboxes_int[i, 0] + bboxes_int[i, 2], bboxes_int[i, 1] + bboxes_int[i, 3]),
                          color_random2[0],
                          2)
            cv2.putText(tempimg, f'{i + 1}th Object_{class_ids[i]}',
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
        height, width = self.GetImageShape(image_id)
        class_ids = []
        for anno in annos:
            if anno['image_id'] == image_id:
                segms.append(np.reshape(anno['segmentation'][0], newshape=(-1, 2)))
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
        self.file = file
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
        count = 0
        self.category2sparse_onehot = {}
        for category in self.categories:
            if category['id'] not in self.category2sparse_onehot:
                self.category2sparse_onehot[category['id']] = count
                count += 1

    def GetOriginalImage(self, image_id):
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
                bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]
                Bboxes.append(bbox)
        return Bboxes

    def GetOriginalCategorySparseList(self, image_id):
        CategoriesSparse = []
        for anno in self.annotations:
            if anno['image_id'] == image_id:
                Sparse = self.category2sparse_onehot[anno['category_id']]
                CategoriesSparse.append(Sparse)
        return CategoriesSparse


    def GetOriginalSegmsMaskList(self, image_id):
        # TODO: put mask list to dictionary of labels
        height, width = self.GetImageShape(image_id)
        Masks = []
        for anno in self.annotations:
            if anno['image_id'] == image_id:
                img_temp = np.zeros(shape=(height, width), dtype=np.uint8)
                contour = np.reshape(anno['segmentation'], newshape=(-1, 2)).astype(int)
                img_temp = cv2.fillPoly(img_temp, [contour], 1)
                Masks.append(img_temp)
        return Masks

    def GetImageName(self, image_id):
        for image in self.images:
            if image['id'] == image_id:
                return image['file_name']

    def GetImageShape(self, image_id):
        for image in self.images:
            if image['id'] == image_id:
                return (image['height'], image['width'])

    def DrawWithImageID(self, image_id):
        original_image = self.GetOriginalImage(image_id)
        self.DrawSegmFromAnnoCoco(image_id, original_image, self.annotations, True)

    def AgumentationOneImage(self, image_id):
        annotation_ids = []
        for anno in self.annotations:
            annotation_ids.append(int(anno['id']))

        max_annotation_id = max(annotation_ids)
        counter = 1
        masks, class_ids = self.GetSegmMaskFromAnnoCOCO(self.annotations, image_id)
        masks = masks.astype(np.uint8)
        img = self.GetOriginalImage(image_id)
        _, _, n_masks = masks.shape
        image_dict = {}
        for img_dic in self.images:
            if img_dic['id'] == image_id:
                image_dict = deepcopy(img_dic)
                break
        # === flip vertically ===
        img_flipped_vertical = np.flip(img, axis=0)
        open_cv_image = cv2.cvtColor(img_flipped_vertical, cv2.COLOR_RGB2BGR)
        image_id_new = f"{image_id}Vertical"
        cv2.imwrite(filename=f"{self.imagefolder_path}/{image_id_new}.png", img=open_cv_image)
        image_dict['id'] = f"{image_id_new}"
        image_dict['file_name'] = f"{image_id_new}.png"
        print(f"image_dic: {image_dict}")
        self.images.append(deepcopy(image_dict))
        for index in range(n_masks):
            mask = masks[:, :, index]
            mask = np.flip(mask, axis=0)
            category_info = {"id": class_ids[index], "is_crowd": False}
            anno = pycococreatortools.create_annotation_info(annotation_id=max_annotation_id + counter,
                                                             image_id=f"{image_id_new}",
                                                             category_info=category_info,
                                                             binary_mask=mask.astype(np.uint8),
                                                             image_size=None,
                                                             tolerance=2,
                                                             bounding_box=None)
            counter += 1
            print(anno)
            self.annotations.append(anno)
        # === flip horizontally ===
        image_id_new = f"{image_id}Horizontal"
        image_dict['id'] = f"{image_id_new}"
        image_dict['file_name'] = f"{image_id_new}.png"
        print(f"image_dic: {image_dict}")
        self.images.append(deepcopy(image_dict))
        img_flipped_horizontal = np.flip(img, axis=1)
        open_cv_image = cv2.cvtColor(img_flipped_horizontal, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename=f"{self.imagefolder_path}/{image_id_new}.png", img=open_cv_image)
        for index in range(n_masks):
            mask = masks[:, :, index]
            mask = np.flip(mask, axis=1)
            category_info = {"id": class_ids[index], "is_crowd": False}
            anno = pycococreatortools.create_annotation_info(annotation_id=max_annotation_id + counter,
                                                             image_id=f"{image_id_new}",
                                                             category_info=category_info,
                                                             binary_mask=mask.astype(np.uint8),
                                                             image_size=None,
                                                             tolerance=2,
                                                             bounding_box=None)
            counter += 1
            print(anno)
            self.annotations.append(anno)
        # === flip both directions ===
        image_id_new = f"{image_id}Both"
        image_dict['id'] = f"{image_id_new}"
        image_dict['file_name'] = f"{image_id_new}.png"
        print(f"image_dic: {image_dict}")
        self.images.append(deepcopy(image_dict))
        img_flipped_both = np.flip(img, axis=(0, 1))
        open_cv_image = cv2.cvtColor(img_flipped_both, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename=f"{self.imagefolder_path}/{image_id_new}.png", img=open_cv_image)
        for index in range(n_masks):
            mask = masks[:, :, index]
            mask = np.flip(mask, axis=(0, 1))
            category_info = {"id": class_ids[index], "is_crowd": False}
            anno = pycococreatortools.create_annotation_info(annotation_id=max_annotation_id + counter,
                                                             image_id=f"{image_id_new}",
                                                             category_info=category_info,
                                                             binary_mask=mask.astype(np.uint8),
                                                             image_size=None,
                                                             tolerance=2,
                                                             bounding_box=None)
            counter += 1
            print(anno)
            self.annotations.append(anno)

    def Augmentation(self):
        if 'augmented' in self.info:
            print('already augmented')
            pass
        else:
            print('start augmenting')
            for image_id in self.image_ids:
                self.AgumentationOneImage(image_id)
            self.info['augmented'] = 'yes'
            with open(self.file, 'w') as f:
                json.dump({
                    "info": self.info,
                    "licenses": self.licenses,
                    "images": self.images,
                    "annotations": self.annotations,
                    "categories": self.categories,
                    "segment_info": self.segment_info
                },
                    f,
                    indent=4)


if __name__ == '__main__':
    file = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/data/1940091026744/annotations/train.json'
    image_path = '/Users/shuzhiliu/Google Drive/KyoceraRobotAI/mmdetection_tools/LocalData_Images'
    t1 = coco_tools(file, image_path)
    t1.DrawWithImageID(t1.image_ids[0])
