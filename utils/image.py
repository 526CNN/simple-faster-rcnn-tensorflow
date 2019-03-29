# -*- coding: utf-8 -*-
"""This module contains tools of image processing.
"""
# TODO: img_scale, multi-input

import os

import cv2
import numpy as np

from model.config import cfg


def image_read(name_image="dat.jpg"):
    """Read image from file and convert into network input.

    Return processed image and scale.
    """
    path_cur = os.path.dirname(__file__)
    path_par = os.path.split(path_cur)[0]
    path = os.path.join(path_par, "misc", name_image)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img_resized, _ = _image_resize(img)
    # _image_show(img_resized)

    return img_resized


def _image_resize(img):
    img_shape = img.shape  # (375, 500, 3)
    img_size_min = np.min(img_shape[0:2])
    img_size_max = np.max(img_shape[0:2])

    img_scale = float(cfg.TARGET_SIZE) / float(img_size_min)  # try scale with small edge
    if np.round(img_scale * img_size_max) > cfg.MAX_SIZE:  # min edge scaling failed, large side should fit
        img_scale = float(cfg.MAX_SIZE) / float(img_size_max)
    img_resized = cv2.resize(img, None, None, fx=img_scale, fy=img_scale,
                             interpolation=cv2.INTER_LINEAR)

    # assert img_resized.shape == (600, 800, 3)

    return img_resized, img_scale


def _image_show(img):
    """This function tests processing of cv2.
    """
    cv2.imshow("SampleWindow", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
