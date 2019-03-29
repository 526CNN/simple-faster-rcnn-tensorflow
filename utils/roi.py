# -*- coding: utf-8 -*-


import tensorflow as tf

from model.config import cfg


def roi_pool(images, boxes):  # images [batch, image_height, image_width, depth]
    size = cfg.ROI_SIZE

    # box_ind refers (i-th boxes)->(box_ind[i])->(images[batch])
    return tf.image.crop_and_resize(images, boxes, [0] * boxes.shape[1], crop_size=size)
           # [num_boxes, crop_height, crop_width, depth]
