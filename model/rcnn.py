# -*- coding: utf-8 -*-


import os

import tensorflow as tf

from base.network import Network
from config import cfg
from utils.rcnn_target_proposal import rcnn_target_proposal
from utils.roi import roi_pool
from utils.anchors import anchor_regress


class RCNN(Network):
    def __init__(self, name_npy="vgg16.npy", image, feature, rpn_boxes):
        path_cur = os.path.dirname(__file__)
        path_par = os.path.split(path_cur)[0]
        path = os.path.join(path_par, "data", name_npy)
        self._data = np.load(path, encoding='latin1').item()

        self._image = image
        self._feature = feature
        self._rpn_boxes = rpn_boxes  # train [2000, 4] inference [300, 4]
        self._rcnn_batch = None
        self._rois = None

        # tf.cond(cfg.TRAIN.IS_TRAIN, self._target_generate())
        self._target_generate()
        self._rcnn_batch = cfg.TRAIN.ROI_BATCH_SIZE

    def build(self):
        self._rois = tf.cond(cfg.TRAIN.IS_TRAIN,
                             roi_pool(self._feature, self._rois),
                             roi_pool(self._feature, self._rpn_boxes))

        self._fc6 = self._fc(_self._rois, "fc6")
        self._fc7 = self._fc(_self._fc6, "fc7")
        self._rcnn_cls = self._rcnn_fc_cls(self._fc7)  # train (128, 21)
        self._rcnn_reg = self._rcnn_fc_reg(self._fc7)  # train (128, 84)

    def loss(self):
        pred_score = self._rcnn_fc_cls  # [128, 21], one hot
        pred_reg = self._rcnn_fc_reg

        gt_score = self._roi_labels  # [128] TODO: to one hot
        gt_reg = self._target_regress
        gt_reg_mask = self._rcnn_reg_mask

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_score,
                                                                                      labels=gt_score))
        smooth_l1 = self._smooth_l1(pred_reg, gt_reg, gt_reg_mask)

        return cross_entropy + smooth_l1


    def inference(self):
        """Yank all needed nodes in computation graph, not just R-CNN.
        """
        rcnn_cls = tf.nn.softmax(self._rcnn_cls, axis=1)
        rcnn_cls_argmax = tf.argmax(rcnn_cls, axis=1)
        rcnn_cls_pred = tf.gather(db.classes, rcnn_cls_argmax)

        boxes_start = tf.multiply(rcnn_cls_argmax, 4)
        boxes_end = tf.add(boxes_start, 4)

        rcnn_reg_pred = self._rcnn_reg[:, boxes_start:boxes_end]
        rcnn_bboxes = anchor_regress(self._rois, rcnn_reg_pred)

        return rcnn_cls_pred, rcnn_bboxes

    def _target_generate(self):
        self._rois, self._roi_labels, self._target_regress, self._rcnn_reg_mask = \
        tf.py_func(rcnn_target_proposal,
                   [self._rpn_boxes],
                   [tf.float32, tf.float32, tf.float32, tf.float32])

    def _rcnn_fc_reg(self, input):  # train input (128, 4096)
        with tf.variable_scope("rcnn_fc_reg", reuse=tf.AUTO_REUSE) as scope:
            weight = tf.get_variable("weight",
                                     initializer=tf.random_normal([4096, cfg.NUM_CLASSES * 4]))
            biases = tf.get_variable("biases",
                                     initializer=tf.zeros([4096, cfg.NUM_CLASSES * 4]))
            fc_biases = self._broadcast_matmul(input, weight, biases)

            return fc_biases


    def _rcnn_fc_cls(self, input):  # input (128, 4096)
        with tf.variable_scope("rcnn_fc_cls", reuse=tf.AUTO_REUSE) as scope:
            weight = tf.get_variable("weight",
                                     initializer=tf.random_normal([4096, cfg.NUM_CLASSES]))
            biases = tf.get_variable("biases",
                                     initializer=tf.zeros([4096, cfg.NUM_CLASSES]))
            fc_biases = self._broadcast_matmul(input, weight, biases)  # [128, 21]

            return fc_biases

    def _fc(self, input, name):  # train input (128, 7, 7, 512) or (128, 4096)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            weight = tf.get_variable("weight", initializer=self._data[name][0])
            biases = tf.get_variable("biases", initializer=self._data[name][1])
            input_reshaped = tf.reshape(input, [self._rcnn_batch, -1])

            fc_biases = self._broadcast_matmul(input_reshaped, weight, biases)
            fc_relu = tf.nn.relu(fc_biases)

            return fc_relu

    def _broadcast_matmul(input, weight, biases):
        with tf.variable_scope("broadcast_matmul"):
            weight_tiled = tf.tile(weight, self._rcnn_batch)
            biases_tiled = tf.tile(biases, self._rcnn_batch)
            fc = tf.matmul(input, weight_tiled)
            fc_biases = tf.nn.bias_add(fc, biases_tiled)

            return fc_biases

    def _smooth_l1(self, pred_reg, gt_reg, gt_reg_mask, sigma=1.0):
        sigma_2 = sigma ** 2
        reg_diff = pred_reg - gt_reg
        reg_diff_filt = tf.multiply(reg_diff, gt_reg_mask)
        reg_diff_filt_abs = tf.abs(reg_diff_filt)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(reg_diff_filt_abs,
                                                             1. / sigma_2)))
        losses_box = tf.pow(reg_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (reg_diff_filt_abs - (0.5 / sigma_2)) * (1. - smoothL1_sign)

        reg_loss = tf.reduce_mean(tf.reduce_sum(losses_box, axis=1))

        return reg_loss
