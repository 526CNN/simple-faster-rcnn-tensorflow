# -*- coding: utf-8 -*-
"""This module will be imported to computation graph with tf.py_func().
"""


import numpy as np

import data.dataset as db
from model.config import cfg
from anchors import anchor_generate
from iou import overlap
from boxes import regress


def rpn_target_proposal(image_h, image_w):
    gt_boxes = db.gt_boxes  # [num_of_gt, 4]
    gt_classes = db.gt_boxes  # [num_of_gt]
    POS_NUM = cfg.TRAIN.RPN_BATCH_SIZE * cfg.TRAIN.RPN_POS_RATIO  # 128


    all_anchors = anchor_generate(image_h, image_w)
    inside_indices = np.where(all_anchors[:, 0] >= 0
                              & all_anchors[:, 1] >= 0
                              & all_anchors[:, 2] <= image_h
                              & all_anchors[:, 3] <= image_w)[0]  # ~ [8000]

    anchors = all_anchors[inside_indices]  # ~ [8000, 4]
    ious = overlap(anchors, gt_boxes)

    anchors_max_ious_arg = ious.argmax(axis=1)  # ~ [0, 0, 1, 0, 1, 0...]
    anchors_best_arg = ious.argmax(axis=0)  # ~  [6253, 2442]
    anchors_max_ious = ious[:, anchors_max_ious_arg]  # ~ [ious, ious, ...]
    gt_anchors = gt_boxes[anchors_max_ious_arg]  # ~ [[y1, x1, y2, x2], ...]

    label = np.zeros(anchors_max_ious.shape[0])
    label.fill(-1)

    label[anchors_best_arg] = 1
    label[anchors_max_ious_arg > cfg.TRAIN.RPN_POS_THRESH] = 1

    label[anchors_max_ious_arg <= cfg.TRAIN.RPN_NEG_THRESH] = 0

    pos_indices = np.where(label == 1)[0]
    pos_num = POS_NUM
    if pos_indices.size > POS_NUM:
        disable_indices = np.random.choice(pos_indices, size=(pos_indices.size - POS_NUM),
                                           replace=False)
        label[disable_indices] = -1

    neg_num = cfg.TRAIN.RPN_BATCH_SIZE - np.sum(label == 1)
    neg_indice = np.where(label == 0)[0]
    if neg_indice.size > neg_num:
        disable_indices = np.random.choice(neg_indice, size=(neg_indice.size - neg_num),
                                           replace=False)
        label[disable_indices] = -1

    target_regress = regress(anchors, gt_anchors)

    rpn_locs_mask = np.zeros(len(inside_indices), 4)  # mask for only pos loss
    rpn_locs_uni = np.zeros(len(inside_indices), 4)  # mask for uniform of boxes loss

    rpn_locs_mask[label == 1, :] = np.array([1., 1., 1., 1.])
    num_pos_to_neg = np.sum(label >= 0)

    rpn_locs_uni[label >= 0, :] = np.array([1. / num_pos_to_neg,
                                            1. / num_pos_to_neg,
                                            1. / num_pos_to_neg,
                                            1. / num_pos_to_neg])

    label, target_regress, rpn_locs_mask, rpn_locs_uni = _extend(label,
                                                                 target_regress,
                                                                 rpn_locs_mask,
                                                                 rpn_locs_uni,
                                                                 all_anchors
                                                                 inside_indices)

    # label = label.reshape([1, image_h, image_w, -1])
    # target_regress = target_regress.reshape([1, image_h, image_w, -1])
    # rpn_locs_mask = rpn_locs_mask.reshape([1, image_h, image_w, -1])
    # rpn_locs_uni = rpn_locs_uni.reshape([1, image_h, image_w, -1])

    return label, target_regress, rpn_locs_mask, rpn_locs_uni


def _extend(label, target_regress, rpn_locs_mask, rpn_locs_uni, all_anchors, inside_indices):
    _label = np.empty(all_anchors.shape[0]).fill(-1)
    _label[inside_indices] = label

    _target_regress = np.empty(all_anchors.shape).fill(0)
    _target_regress[inside_indices] = target_regress

    _rpn_locs_mask = np.empty(all_anchors.shape).fill(0)
    _rpn_locs_mask[inside_indices] = rpn_locs_mask

    _rpn_locs_uni = np.empty(all_anchors.shape).fill(0)
    _rpn_locs_uni[inside_indices] = rpn_locs_uni

    return _label, _target_regress, _rpn_locs_mask, _rpn_locs_uni
