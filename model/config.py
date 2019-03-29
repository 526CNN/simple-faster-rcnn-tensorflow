# -*- coding: utf-8 -*-
"""This file use easydict as various params of Faster R-CNN,
as a way to access with attribute.
"""


from easydict import EasyDict


# Set with _C
_C = EasyDict()

# Export with cfg
cfg = _C


# Params of Faster R-CNN model
_C.VGG_MEAN = [103.939, 116.779, 123.68]
_C.MAX_SIZE = 1000
_C.TARGET_SIZE = 600
_C.SUB_SAMPLE = 16
_C.NMS_THRESH = 0.7
_C.ANCHOR_NUM = 9
_C.ANCHOR_RATIO = [0.5, 1, 2]
_C.ANCHOR_SIZE = [8, 16, 32]
_C.NUM_CLASSES = 20 + 1  # background
_C.ROI_SIZE = (7, 7)

# Params of Train
_C.TRAIN.IS_TRAIN = True
_C.TRAIN.NMS_PRE_TOPN = 12000
_C.TRAIN.NMS_POST_TOPN = 2000
_C.TRAIN.RPN_BATCH_SIZE = 256
_C.TRAIN.RPN_POS_RATIO = 0.5
_C.TRAIN.RPN_POS_THRESH = 0.7
_C.TRAIN.RPN_NEG_THRESH = 0.3
_C.TRAIN.ROI_POS_THRESH = 0.5
_C.TRAIN.ROI_NEG_THRESH_HI = 0.5
_C.TRAIN.ROI_NEG_THRESH_LO = 0.0
_C.TRAIN.ROI_BATCH_SIZE = 128
_C.TRAIN.ROI_POS_RATIO = 0.25

# Params of Inference
