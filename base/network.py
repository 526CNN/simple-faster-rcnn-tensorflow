# -*- coding: utf-8 -*-
"""This base class demonstrate the bahaviour that models should follow.
"""


class Network(object):
    def __init__(self):
        pass

    def build(self):
        raise NotImplementedError

    def _target_generate(self):
        """Deal with ground truth to produce calculate loss.

        Actually import utils, this is an encapsulation with tf.py_func to ops.

        For RPN: with anchors and gt boxes, each cycle, then apply:
            1. Reduce anchors size (from generator) use crop outside.
            2. Calculate IoU between each anchor and gt, get [num_anchors, num_gt].
            3. Argsort to select best gt to each anchor,
               as [0, 3, 1, 2...] (length is num of anchors, inside which is n-th gt box),
               assgin 1D anchors with that IoU.
            4. Use 1D anchor and threshold to assgin (pos/neg/ignore) a same shape 1D label, as [0, 1, 1, 0, -1...].
            5. Calculate every target regress for 1D anchors with their best iou gt boxes.
            6. In regard of ONLY pos loss, create weight mask for locs/cls anchors,
                (1). Locs mask, randomly choice 256 * 0.5 positive anchors, assign with 1, other 0,
                    [[0, 0, 0, 0],
                     [1, 1, 1, 1],
                     [0, 0, 0, 0],
                         ...
                     [0, 0, 0, 0]].
                (2). Cls mask, all 1 but divide with sum of label >= 0 (all valid anchors 1/sum, other 0).
            7. Extend the shape to num_anchors (optionally if do all with index) for all.
            8. Return label, target_regress, rpn_locs_mask, rpn_locs_uni (mask of all anchors).

        For R-CNN: with RPN boxes process inside, each cycle, apply them to rpn_reg_locs, gt_classes, gt_locs:
            1. replace cropped_anchors with rpn_reg_locs,
               replace threshold for fg/bg,
               random choice 128.
            2. do RPN(2-6), label should be 0 or 1.
            3. Reture rpn_locs_regressed_selected, label, target_regress,
               rcnn_reg_mask (only not 0 classes, expanded with classes, like RPN inside mask).

        Note that:
        RPN target meets the shape of RPN output,
        R-CNN target meets the shape of RoI input (also R-CNN output).

        rpn_locs_mask plays a role of weight filter for rpn locs loss,
        rpn_locs_uni acts like mean of pos/neg loss.

        rcnn_reg_mask performs a usage of weight filter for only corrent classes.
        """
        raise NotImplementedError
