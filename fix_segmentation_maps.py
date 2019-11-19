import numpy as np
import os
import cv2
import copy

def _fix(data_dir):
    color_list = [[142, 0, 0], [60, 20, 220], [32, 11, 119]]
    img_seg_onehot = []

    for fn in sorted(os.listdir(data_dir)):
        file_ = os.path.join(data_dir, fn)
        file_arr = cv2.imread(file_)
        for color in color_list:
            equality = np.equal(file_arr, color)
            class_map = np.all(equality, axis=-1)
            img_seg_onehot.append(class_map)
        img_seg_onehot_dup = copy.deepcopy(img_seg_onehot)
        img_seg_onehot_dup = np.array(img_seg_onehot_dup)
        background_cls = np.any(img_seg_onehot_dup, axis= -1)
        background_cls[background_cls==0] = 1
        img_seg_onehot.append(background_cls)
        img_seg_onehot = np.stack(img_seg_onehot, axis=-1)

