# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf

from model.config import cfg


def anchor_generate(img_h, img_w):
    h = img_h // cfg.SUB_SAMPLE
    w = img_w // cfg.SUB_SAMPLE

    ratio = cfg.ANCHOR_RATIO
    size = cfg.ANCHOR_SIZE

    ctr_x = np.arange(16, (w + 1) * 16, 16)
    ctr_y = np.arange(16, (h + 1) * 16, 16)

    ctr = np.zeros((len(ctr_x) * len(ctr_y), 2), dtype=np.float32)
    index = 0
    for i in range(len(ctr_x)):
        for j in range(len(ctr_y)):
            ctr[index, 1] = ctr_x[i] - 8
            ctr[index, 0] = ctr_y[j] - 8
            index += 1

    anchors = np.zeros((h * w
                       * len(ratio)
                       * len(size), 4),
                       dtype=np.float32)

    index = 0
    for c in ctr:
        ctr_y, ctr_x = c
        for i in range(len(ratio)):
            for j in range(len(size)):
                h = cfg.SUB_SAMPLE * ratio[j] * np.sqrt(ratio[i])
                w = cfg.SUB_SAMPLE * ratio[j] * np.sqrt(1. / ratio[i])

                anchors[index, 0] = ctr_y - h / 2
                anchors[index, 1] = ctr_x - w / 2
                anchors[index, 2] = ctr_y + h / 2
                anchors[index, 3] = ctr_x + w / 2
                index += 1

    return anchors  # (512 * h * w, 4) y1, x1, y2, x2


def anchor_regress(anchors, regs):
    ctr_y = anchors[:, 0]
    ctr_x = anchors[:, 1]
    h = anchors[:, 2]
    w = anchors[:, 3]

    dy = regs[:, 0]
    dx = regs[:, 1]
    dh = regs[:, 2]
    dw = regs[:, 3]

    ctr_y_reg = tf.add(tf.multiply(dy, h), ctr_y)
    ctr_x_reg = tf.add(tf.multiply(dx, w), ctr_x)
    h_reg = tf.multiply(h, tf.exp(dh))
    w_reg = tf.multiply(w, tf.exp(dw))

    boxes_ctr = tf.stack([ctr_y_reg, ctr_x_reg, h_reg, w_reg], axis=1)

    return boxes_ctr
