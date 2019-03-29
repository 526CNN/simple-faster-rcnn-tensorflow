# -*- coding: utf-8 -*-
"""This module process on original image.
"""

import numpy as np

import data.dataset as db
from model.config import cfg
from iou import overlap
from boxes import to_box_ctr_np, regress


def rcnn_target_proposal(rpn_boxes):  # train [2000, 4]
    """This function choose suitable boxes from rpn_boxes in training time,
    which will then be feeded into RoI as a batch (batch size 128 in training).

    In inference time, rpn_boxes deduct inside rpn._rpn_proposal to 300.
    """
    gt_boxes = db.gt_boxes  # [num_of_gt, 4]
    gt_classes = db.gt_boxes  # [num_of_gt]
    POS_NUM = cfg.TRAIN.ROI_BATCH_SIZE * TRAIN.ROI_POS_RATIO  # 32

    ious = overlap(rpn_boxes, gt_boxes)
    boxes_max_ious_arg = ious.argmax(axis=1)
    boxes_max_ious = ious[:, boxes_max_ious_arg]

    gt_labels = gt_classes[boxes_max_ious_arg]
    gt_boxes = gt_boxes[boxes_max_ious_arg]

    pos_indices = np.where(boxes_max_ious >= cfg.TRAIN.ROI_POS_THRESH)[0]
    pos_num = int(min(POS_NUM, pos_indices.size))  # size on this batch
    if pos_indices.size > 0:
        pos_indices = np.random.choice(pos_indices, size=pos_num, replace=False)  # False: no repeat

    neg_indices = np.where(boxes_max_ious >= cfg.TRAIN.ROI_NEG_THRESH_LO and \
                           boxes_max_ious < cfg.TRAIN.ROI_NEG_THRESH_HI)[0]
    neg_num = cfg.TRAIN.ROI_BATCH_SIZE - pos_num
    if neg_indices.size > 0:
        neg_indices = np.random.choice(neg_indices, size=neg_num, replace=False)

    keep_indices = np.append(pos_indices, neg_indices)

    gt_labels = gt_labels[keep_indices]
    gt_boxes = gt_boxes[keep_indices]
    gt_labels[pos_num:] = 0  # negative sampel assgin to background
    roi_labels = gt_labels  # rename
    rois = rpn_boxes[keep_indices]  # feed forward, R-CNN

    reg = regress(rois, gt_boxes)  # [128, 4]
    target_regress = _expand(reg, roi_labels)  # [128, 84], expand for rcnn._rcnn_fc_reg in training
    rcnn_reg_mask = _make_mask(gt_labels)  # [128, 84]

    return rois, roi_labels, target_regress, rcnn_reg_mask
    #   [128, 4]   [128]       [128, 84]       [128, 84]


def _expand(reg, label):
    reg_expanded = np.zeros([cfg.TRAIN.ROI_BATCH_SIZE, 4 * cfg.NUM_CLASSES])  # [128, 84]
    reg_expanded[:, label*4 : label*4 + 4] = reg

    return reg_expanded


def _make_mask(label):
    mask = np.zeros([cfg.TRAIN.ROI_BATCH_SIZE, 4 * cfg.NUM_CLASSES])  # [128, 84]
    mask[:, label*4 : label*4 + 4] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    return mask
