from NN_Helper import GenBaseAnchors
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
from Debugger import debug_print


class GenCandidateAnchors:
    def __init__(self,
                 base_size=16,
                 ratios=None,
                 scales=2 ** np.arange(3, 6),
                 img_shape=(800, 1333, 3),
                 n_stage=5,
                 n_anchors=9):
        if ratios is None:
            ratios = [0.5, 1, 2]
        self.img_shape = (img_shape[0], img_shape[1])
        self.n_stage_revert_factor = 2 ** n_stage
        self.base_anchors = GenBaseAnchors.gen_base_anchors(base_size=base_size, ratios=ratios,
                                                            scales=scales)
        # here round up the number since the tensorflow conv2d round strategy
        self.h = int(img_shape[0] / self.n_stage_revert_factor) + int((img_shape[0] % self.n_stage_revert_factor) > 0)
        self.w = int(img_shape[1] / self.n_stage_revert_factor) + int((img_shape[1] % self.n_stage_revert_factor) > 0)
        self.n_anchors = n_anchors
        self.anchor_candidates = self.gen_all_candidate_anchors(self.h, self.w, self.n_anchors, self.img_shape)
        self.anchor_candidates_list = list(np.reshape(self.anchor_candidates, newshape=(-1, 4)).tolist())

    def gen_all_candidate_anchors(self, h, w, num_anchors, image_shape):
        anchors = np.zeros(shape=(h, w, num_anchors, 4), dtype=np.int)
        # anchors axis format: (x1, y1, x2, y2)
        x_max = image_shape[0] - 1
        y_max = image_shape[1] - 1
        for x in range(h):
            for y in range(w):
                temp = self.base_anchors + np.array([x * self.n_stage_revert_factor - self.n_stage_revert_factor / 2,
                                                     y * self.n_stage_revert_factor - self.n_stage_revert_factor / 2,
                                                     x * self.n_stage_revert_factor - self.n_stage_revert_factor / 2,
                                                     y * self.n_stage_revert_factor - self.n_stage_revert_factor / 2])
                temp[:, 0][temp[:, 0] < 0] = 0
                temp[:, 1][temp[:, 1] < 0] = 0
                temp[:, 2][temp[:, 2] > x_max] = x_max
                temp[:, 3][temp[:, 3] > y_max] = y_max
                anchors[x, y, :, :] = temp

        return anchors

    def _validate_bbox(self, bboxes):
        img1 = np.zeros(shape=(self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8)
        for bbox in bboxes:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # cv.rectangle(img1,(bbox[0],bbox[1]), (bbox[2],bbox[3]), 255)
            # print(bbox)
            cv.rectangle(img=img1, rec=(bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]), color=color,
                         thickness=4)
        plt.imshow(img1)
        plt.show()


def get_feature_map_h_w_with_n_stages(img_shape, n_stage):
    # here round up the number since the tensorflow conv2d round strategy
    n_stage_revert_factor = 2 ** n_stage
    h = int(img_shape[0] / n_stage_revert_factor) + int((img_shape[0] % n_stage_revert_factor) > 0)
    w = int(img_shape[1] / n_stage_revert_factor) + int((img_shape[1] % n_stage_revert_factor) > 0)
    return h, w


if __name__ == '__main__':
    t1 = GenCandidateAnchors(base_size=12)
    debug_print("base anchors", t1.base_anchors)
    debug_print("9 Anchors candidate at [0,0]", t1.anchor_candidates[0, 0, :, :])
    debug_print("9 Anchors candidate at [10,10]", t1.anchor_candidates[10, 10, :, :])
    # bboxes = [t1.anchors_candidate[10, 10, i, :] for i in range(9)]
    bboxes = t1.anchor_candidates[13, 22, :, :].tolist()
    t1._validate_bbox(bboxes)
    # print(t1.all_anchors[23,40,:,:])
    debug_print("1 Anchor candidate at [0,0,2]", t1.anchor_candidates[0, 0, 2, :])
    # temp = t1.anchors_candidate.reshape((-1, 4))
    temp = np.reshape(t1.anchor_candidates, newshape=(-1, 4))
    debug_print("Same with Anchor candidate at [0,0,2] after reshape", temp[2, :])
    temp2 = temp.reshape((t1.h, t1.w, t1.n_anchors, 4))
    debug_print("Same with temp[2,:] after reshape", temp2[0, 0, 2, :])
    debug_print("Same anchor in anchor list", t1.anchor_candidates_list[2])
