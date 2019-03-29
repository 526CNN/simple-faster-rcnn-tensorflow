# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


def to_box_cor(boxes):
    """Convert (ctr_y, ctr_x, h, w) to (y1, x1, y2, x2)
    """

    ctr_y = boxes[:, 0]
    ctr_x = boxes[:, 1]
    h = boxes[:, 2]
    w = boxes[:, 3]

    y1 = tf.subtract(ctr_y, tf.multiply(.5, h))
    x1 = tf.subtract(ctr_x, tf.multiply(.5, w))
    y2 = tf.add(ctr_y, tf.multiply(.5, h))
    x2 = tf.add(ctr_x, tf.multiply(.5, w))

    boxes_cor = tf.stack([y1, x1, y2, x2], axis=1)  # equal to (axis=0).transpose()

    return boxes_cor


def to_box_ctr(boxes):
    """Convert (y1, x1, y2, x2) to (ctr_y, ctr_x, h, w)
    """
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    h = tf.subtract(y2, y1)
    w = tf.subtract(x2, x1)
    ctr_y = tf.add(y1, tf.multiply(h, .5))
    ctr_x = tf.add(x1, tf.multiply(h, .5))

    boxes_ctr = tf.stack([ctr_y, ctr_x, h, w], axis=1)

    return boxes_ctr


def box_cropper(boxes, h, w):
    # check_left_top(check_right_bottom)
    y1 = tf.maximum(tf.minimum(boxes[:, 0], h - 1), 0)
    x1 = tf.maximum(tf.minimum(boxes[:, 1], w - 1), 0)
    y2 = tf.maximum(tf.minimum(boxes[:, 2], h - 1), 0)
    x2 = tf.maximum(tf.minimum(boxes[:, 3], w - 1), 0)
    box_cropped = tf.stack([y1, x1, y2, x2], axis=1)

    return box_cropped


def to_box_ctr_np(boxes):
    """This function transform the boxes' format.

    With (y1, x1, y2, x2) to (ctr_y, ctr_x, h, w).
    """
    h = boxes[:, 2] - boxes[:, 0]
    w = boxes[:, 3] - boxes[:, 1]
    ctr_y = boxes[:, 0] + 0.5 * h
    ctr_x = boxes[:, 1] + 0.5 * w

    boxes = np.vstack((ctr_y, ctr_x, h, w)).transpose()

    return boxes


def to_box_cor_np(boxes):
    """This function transform the boxes' format.

    With (ctr_y, ctr_x, h, w) to (y1, x1, y2, x2).
    """
    y1 = boxes[:, 0] - boxes[:, 2] * 0.5
    x1 = boxes[:, 1] - boxes[:, 3] * 0.5
    y2 = boxes[:, 0] + boxes[:, 2] * 0.5
    x2 = boxes[:, 1] + boxes[:, 3] * 0.5

    boxes = np.vstack((y1, x1, y2, x2)).transpose()

    return boxes


def regress(locs, locs_gt):
    locs_ctr = to_box_ctr_np(locs)  # ctr_y, ctr_x, h, w
    locs_gt_ctr = to_box_ctr_np(locs_gt)

    ctr_y = locs_ctr[:, 0]
    ctr_x = locs_ctr[:, 1]
    h = locs_ctr[:, 2]
    w = locs_ctr[:, 3]

    # prevent 0 divide
    eps = np.finfo(np.float32).eps
    h = np.maximum(h, eps)
    w = np.maximum(w, eps)

    ctr_y_gt = locs_gt_ctr[:, 0]
    ctr_x_gt = locs_gt_ctr[:, 1]
    h_gt = locs_gt_ctr[:, 2]
    w_gt = locs_gt_ctr[:, 3]

    dy = (ctr_y_gt - ctr_y) / h
    dx = (ctr_x_gt - ctr_x) / w
    dh = np.log(h_gt / h)
    dw = np.log(w_gt / w)

    reg = np.vstack((dy, dx, dh, dw)).transpose()

    return reg
