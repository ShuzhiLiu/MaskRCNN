import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pycococreatortools import pycococreatortools
from copy import deepcopy


class CocoTools:
    def __init__(self, jsonfile, imagefolder_path, resized_shape=None):
        self.load_anno_coco(jsonfile, imagefolder_path)
        self.RESIZE_FLAG = False
        self.resized_shape = resized_shape
        if resized_shape != None:
            self.RESIZE_FLAG = True

    def load_anno_coco(self, file, imagefolder_path):
        self.imagefolder_path = imagefolder_path
        self.file = file
        with open(file, 'r') as f:
            dict1 = json.load(f)
        self.info = dict1["info"]
        self.licenses = dict1["licenses"]
        self.images = dict1["images"]
        self.annotations = dict1["annotations"]
        self.categories = dict1["categories"]
        # self.segment_info = dict1["segment_info"]
        self.image_ids = []
        for image in self.images:
            if image['id'] not in self.image_ids:
                self.image_ids.append(image['id'])
        count = 0
        self.category2sparse_onehot = {}
        self.sparse_onehot2category = []
        for category in self.categories:
            if category['id'] not in self.category2sparse_onehot:
                self.category2sparse_onehot[category['id']] = count
                self.sparse_onehot2category.append(category['id'])
                count += 1

    def _resize_anno(self):
        for image_id in self.image_ids:
            original_shape = self.get_image_shape(image_id)

    def draw_segm_from_anno_coco(self, image_id, original_image, annos, show=False, savefile=False):
        height, width = self.get_image_shape(image_id)
        if self.RESIZE_FLAG:
            height, width, _ = self.resized_shape
        bboxes_int = self.get_original_bboxes_list(image_id)  # format (x1,y1,x2,y2)
        tempimg = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        masks_pred, class_ids = self.get_segm_mask_from_anno_coco(annos, image_id)
        _, _, n_masks = masks_pred.shape
        for i in range(n_masks):
            color_random = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_random2 = color_random.tolist()
            # opencv need format (y1, x1, y2, x2)
            cv2.rectangle(tempimg,
                          (bboxes_int[i][1], bboxes_int[i][0]),
                          (bboxes_int[i][3], bboxes_int[i][2]),
                          color_random2[0],
                          2)
            cv2.putText(tempimg, f'{i + 1}th Object_{class_ids[i]}',
                        (bboxes_int[i][3] + 10, bboxes_int[i][2]),
                        0, 0.3, color_random2[0])
            tempimg[:, :, 0][masks_pred[:, :, i]] = color_random[0, 0]
            tempimg[:, :, 1][masks_pred[:, :, i]] = color_random[0, 1]
            tempimg[:, :, 2][masks_pred[:, :, i]] = color_random[0, 2]
        original_image = (original_image * 0.5 + tempimg * 0.5).astype(np.uint8)
        plt.imshow(original_image)
        if show:
            plt.show()
        if savefile:
            plt.savefig(f'{os.getcwd()}/Images_Drawn/{image_id}.jpg', dpi=300)

    def draw_bboxes(self, original_image, bboxes, show=False, savefile=False, path=None, savename=None):
        # bbox is numpy format (x1, y1, x2, y2)
        height, width = original_image.shape[0], original_image.shape[1]
        tempimg = np.zeros(shape=(height, width, 3), dtype=np.uint8)
        for bbox in bboxes:
            color_random = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            color_random2 = color_random.tolist()
            cv2.rectangle(tempimg,
                          (bbox[1], bbox[0]),
                          (bbox[3], bbox[2]),
                          color_random2[0],
                          2)
        original_image = (original_image * 0.5 + tempimg * 0.5).astype(np.uint8)
        plt.imshow(original_image)
        if show:
            plt.show()
        if savefile:
            img_opencv = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename=f"{path}/{savename}.jpg", img=img_opencv)

    def get_segm_mask_from_anno_coco(self, annos, image_id):
        segms = []
        height, width = self.get_image_shape(image_id)
        if self.RESIZE_FLAG:
            height, width, _ = self.resized_shape
        class_ids = []
        for anno in annos:
            if anno['image_id'] == image_id and isinstance(anno['segmentation'], list):
                segm_temp = np.reshape(anno['segmentation'][0], newshape=(-1, 2))
                if self.RESIZE_FLAG:
                    original_shape = self.get_image_shape(image_id)
                    # Note that opencv format is (y, x) here, different from numpy (x, y)
                    segm_temp = segm_temp / np.asarray([original_shape[1], original_shape[0]]) * np.asarray(
                        [self.resized_shape[1], self.resized_shape[0]])
                segms.append(segm_temp.astype(int))
                class_ids.append(anno['category_id'])
        n_segms = len(segms)
        mask_temp = np.zeros(shape=(height, width, n_segms), dtype=np.uint8)
        for i in range(n_segms):
            temp_one_mask = np.zeros(shape=(height, width), dtype=np.uint8)
            contour = segms[i]
            cv2.fillPoly(img=temp_one_mask, pts=[contour], color=1)
            mask_temp[:, :, i] = temp_one_mask
        return mask_temp.astype(np.bool), class_ids

    def get_original_image(self, image_id):
        image_name = self.get_image_name(image_id)
        img = cv2.imread(f"{self.imagefolder_path}/{image_name}")
        if self.RESIZE_FLAG:
            img = cv2.resize(img, (self.resized_shape[1], self.resized_shape[0]), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.array(img_rgb)

    def get_original_bboxes_list(self, image_id):
        # read original opencv bbox and convert to numpy format bbox
        '''
        x: vertical, y: horizontal. (x, y) is the format for numpy array
        opencv box format: (y, x, dy, dx)
        output box format: (x1, y1, x2, y2)
        '''
        bboxes = []
        for anno in self.annotations:
            if anno['image_id'] == image_id:
                bbox = np.array(anno['bbox'], dtype=np.int)
                bbox[0], bbox[1], bbox[2], bbox[3] = bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]
                if self.RESIZE_FLAG:
                    original_shape = self.get_image_shape(image_id)
                    bbox[0] = bbox[0] / original_shape[0] * self.resized_shape[0]
                    bbox[1] = bbox[1] / original_shape[1] * self.resized_shape[1]
                    bbox[2] = bbox[2] / original_shape[0] * self.resized_shape[0]
                    bbox[3] = bbox[3] / original_shape[1] * self.resized_shape[1]
                bboxes.append(bbox)
        return bboxes

    def get_original_category_sparse_list(self, image_id):
        categories_sparse = []
        for anno in self.annotations:
            if anno['image_id'] == image_id:
                Sparse = self.category2sparse_onehot[anno['category_id']]
                categories_sparse.append(Sparse)
        return categories_sparse

    def get_original_segms_mask_list(self, image_id):
        # TODO: put mask list to dictionary of labels
        height, width = self.get_image_shape(image_id)
        Masks = []
        for anno in self.annotations:
            if anno['image_id'] == image_id:
                img_temp = np.zeros(shape=(height, width), dtype=np.uint8)
                contour = np.reshape(anno['segmentation'], newshape=(-1, 2)).astype(int)
                img_temp = cv2.fillPoly(img_temp, [contour], 1)
                Masks.append(img_temp)
        return Masks

    def get_image_name(self, image_id):
        for image in self.images:
            if image['id'] == image_id:
                return image['file_name']

    def get_image_shape(self, image_id):
        for image in self.images:
            if image['id'] == image_id:
                return (image['height'], image['width'])

    def draw_with_image_id(self, image_id):
        original_image = self.get_original_image(image_id)
        self.draw_segm_from_anno_coco(image_id, original_image, self.annotations, True)

    def agumentation_one_image(self, image_id):
        annotation_ids = []
        for anno in self.annotations:
            annotation_ids.append(int(anno['id']))

        max_annotation_id = max(annotation_ids)
        counter = 1
        masks, class_ids = self.get_segm_mask_from_anno_coco(self.annotations, image_id)
        masks = masks.astype(np.uint8)
        img = self.get_original_image(image_id)
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

    def augmentation(self):
        if 'augmented' in self.info:
            print('already augmented')
            pass
        else:
            print('start augmenting')
            for image_id in self.image_ids:
                self.agumentation_one_image(image_id)
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

    def make_train_sample(self, n, file):
        images = []
        annotations = []
        images += self.images[:n]
        for image in images:
            image_id = image['id']
            for anno in self.annotations:
                if anno['image_id'] == image_id:
                    annotations.append(anno)
        anno_json = {
            "info": self.info,
            "licenses": self.licenses,
            "images": images,
            "annotations": annotations,
            "categories": self.categories,
            # "segment_info": self.segment_info
        }
        with open(file, 'w') as f:
            json.dump(anno_json, f)


if __name__ == '__main__':
    # file = '/Volumes/HDD500/mmdetection_tools/data/1988605221046/annotations/train.json'
    # image_path = '/Volumes/HDD500//mmdetection_tools/LocalData_Images'
    # file = '/Volumes/HDD500/Dataset/COCO2017/annotations/instances_val2017.json'
    # file_sample = '/Volumes/HDD500/Dataset/COCO2017/annotations/instances_val2017_sample.json'
    # image_path = '/Volumes/HDD500/Dataset/COCO2017/val2017'
    file = '/media/liushuzhi/HDD500/Dataset/COCO2017/annotations/instances_val2017.json'
    file_sample = '/media/liushuzhi/HDD500/Dataset/COCO2017/annotations/instances_val2017_sample.json'
    image_path = '/media/liushuzhi/HDD500/Dataset/COCO2017/val2017'
    t1 = CocoTools(file, image_path, resized_shape=(800, 1333, 3))
    t1.make_train_sample(n=20, file=file_sample)
