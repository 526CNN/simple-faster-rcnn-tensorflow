# -*- coding: utf-8 -*-
"""This is a numpy module, builded for rpn/rcnn target generator in training time,
which will then be imported into graph with tf.py_func().

In inference time, use tf's library nms, there is no need to use this module.
"""


import numpy as np


def overlap(boxes, boxes_gt):  # [2000, 4], [num_of_gt, 4] or [512*h*w, 4], [num_of_gt, 4]
    ious = np.zeros([boxes.shape[0], boxes_gt[0]], dtype=np.float32)

    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    y1_gt, x1_gt, y2_gt, x2_gt = boxes_gt[:, 0], boxes_gt[:, 1], boxes_gt[:, 2], boxes_gt[:, 3]

    area = (x2 - x1) * (y2 - y1)
    area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    for ind in boxes:
        for ind_gt in boxes_gt:
            y1_inner = np.maximum(y1[ind], y1_gt[ind_gt])
            x1_inner = np.maximum(x1[ind], x1_gt[ind_gt])
            y2_inner = np.minimum(y2[ind], y2_gt[ind_gt])
            x2_inner = np.minimum(x2[ind], x2_gt[ind_gt])
            ious[ind][ind_gt] = 0 if not (x2_inner > x1_inner and y2_inner > y1_inner) \
                                else (y2_inner - y1_inner) * (x2_inner - x1_inner) \
                                     / (area[ind] + area_gt[ind_gt]) \
                                     - (y2_inner - y1_inner) * (x2_inner - x1_inner)

    return ious
