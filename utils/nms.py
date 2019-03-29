# -*- coding: utf-8 -*-


import tensorflow as tf


def nms(boxes, scores, max_out, thresh):
    """Apply non maximum suppression

    boxes follow (y1, x1, y2, x2)
    """
    return tf.image.non_max_suppression(boxes, scores, max_output_size=max_out,
                                        iou_threshold=thresh)
