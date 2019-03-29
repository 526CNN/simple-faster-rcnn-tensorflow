# -*- coding: utf-8 -*-


import tensorflow as tf

from config import cfg
from base.network import Network
from utils.nms import nms
from utils.boxes import to_box_cor, box_cropper, to_box_ctr
from utils.anchors import anchor_generate, anchor_regress
from utils.rpn_target_proposal import rpn_target_proposal


class RPN(Network):
    def __init__(self, feature, image):
        self._feature = feature
        self._image = image
        self._h = self._image.shape[0]
        self._w = self._image.shape[1]
        self._h_feat = self._h // cfg.SUB_SAMPLE
        self._w_feat = self._w // cfg.SUB_SAMPLE

    def build(self):
        self._rpn_feature = self._rpn_conv2d(self._feature)
        self._rpn_reg_locs = self._rpn_conv2d_reg(self._rpn_feature)  # [512*_h*_w, 4], input of rpn_target_generate
        self._rpn_cls_score = self._rpn_conv2d_cls(self._rpn_feature)  # [512*_h*_w, 1], input of rpn_target_generate

        self.rpn_bboxes = self._rpn_proposal(self._rpn_reg_locs,  # only bboxes [2000, 4] is given, input of R-CNN
                                             self._rpn_cls_score)

        # tf.cond(cfg.TRAIN.IS_TRAIN, self._target_generate())
        self._target_generate()

    def loss(self):
        pred_reg = self._rpn_reg_locs
        pred_score = self._rpn_cls_score

        gt_reg = self._target_regress
        gt_score = self._labels

        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_score,
                                                                                      labels=gt_score))
        smooth_l1 = _smooth_l1(pred_reg, gt_reg, self._rpn_locs_mask, self._rpn_locs_uni)

        return cross_entropy + smooth_l1

    def _target_generate(self):
        self._labels, self._target_regress, self._rpn_locs_mask, self._rpn_locs_uni = \
        tf.py_func(rpn_target_proposal,
                   [self._h, self._w],
                   tf.float32, tf.float32, tf.float32, tf.float32)

    def _rpn_proposal(self, rpn_reg_locs, rpn_cls_score):
        """Deduction the output of RPN, is input of RoI.

        NMS_pre_TopN -> apply nms -> NMS_post_TopN
        """
        anchors = tf.py_func(anchor_generate,
                             [self._h, self._w],
                             [tf.float32])
        anchors = to_box_ctr(anchors)

        boxes_regressed = anchor_regress(anchors, rpn_reg_locs)  # apply transform to all anchors

        rpn_boxes = to_box_cor(boxes_regressed)
        rpn_score_arg = tf.argsort(rpn_cls_score, direction='DESCENDING')
        rpn_arg_top_pre = rpn_score_arg[:cfg.TRAIN.NMS_PRE_TOPN]

        rpn_boxes_top_pre = tf.gather(rpn_boxes, rpn_arg_top_pre)
        rpn_score_top_pre = tf.gather(rpn_cls_score, rpn_arg_top_pre)
        indices = nms(rpn_boxes_top_pre, rpn_score_top_pre,
                      cfg.TRAIN.NMS_POST_TOPN, cfg.NMS_THRESH)

        rpn_proposal = tf.gather(rpn_boxes_top_pre, indices)
        rpn_proposal_cropped = box_cropper(rpn_proposal, self._h, self._w)

        return rpn_proposal_cropped  # train: [2000, 4] y1, x1, y2, x2


    def _rpn_conv2d(self, input, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            filter = tf.get_variable("filter", initializer=tf.truncated_normal((3, 3, 512, 512), stddev=0.01))
            biases = tf.get_variable("biases", initializer=tf.zeros((3, 3, 512, 512)))
            conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1],
                                padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_relu = tf.nn.relu(conv_bias)

        return conv_relu

    def _rpn_conv2d_cls(self, input, name="rpn_conv2d_cls"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            filter = tf.get_variable("filter", initializer=tf.truncated_normal((1, 1, 512, 18), stddev=0.01))
            biases = tf.get_variable("biases", initializer=tf.zeros((1, 1, 512, 18)))
            conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1],
                                padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_shape = tf.shape(conv_bias)
            conv_reshaped = tf.reshape(conv_bias, [-1, 2])
            conv_softmaxed = tf.nn.softmax(conv_reshaped, axis=1)
            conv_reshape_back = tf.reshape(conv_softmaxed, conv_shape)
            conv_score = conv_reshape_back[:, :, :, cfg.ANCHOR_NUM:]
            conv_score_flatten = tf.reshape(conv_score, [-1])

        return conv_score_flatten

    def _rpn_conv2d_reg(self, input, name="rpn_conv2d_reg"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            filter = tf.get_variable("filter", initializer=tf.truncated_normal((1, 1, 512, 36), stddev=0.01))
            biases = tf.get_variable("biases", initializer=tf.zeros((1, 1, 512, 36)))
            conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1],
                                padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_reg_flatten = tf.shape(conv_bias, [-1, 4])

        return conv_reg_flatten

    def _smooth_l1(self, pred_reg, gt_reg, rpn_locs_mask, rpn_locs_uni, sigma=3.0):
        sigma_2 = sigma ** 2
        reg_diff = pred_reg - gt_reg
        reg_diff_filt = tf.multiply(reg_diff, rpn_locs_mask)
        reg_diff_filt_abs = tf.abs(reg_diff_filt)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(reg_diff_filt_abs,
                                                             1. / sigma_2)))
        losses_box = tf.pow(reg_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (reg_diff_filt_abs - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        losses_box_uni = tf.multiply(losses_box, rpn_locs_uni)

        reg_loss = tf.reduce_mean(tf.reduce_sum(losses_box, axis=1))

        return reg_loss
