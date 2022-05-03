#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "./data/classes/kitti.names"
__C.YOLO.ANCHORS                = "./data/anchors/kitti_train_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.999
__C.YOLO.STRIDES                = [8, 16,32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.6
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo_voc.ckpt"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "./data/dataset/ada_train.txt"
__C.TRAIN.BATCH_SIZE            = 1
__C.TRAIN.INPUT_SIZEX            = [192,224,256]
__C.TRAIN.INPUT_SIZEY            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG              = False
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_coco_demo.ckpt"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "./data/dataset/difficultval.txt"
__C.TEST.BATCH_SIZE             = 1
__C.TEST.INPUT_SIZEX             = 256
__C.TEST.INPUT_SIZEY             =768
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "./cross/cross/model-124000"
__C.TEST.SHOW_LABEL             = False
__C.TEST.SCORE_THRESHOLD        = 0.5
__C.TEST.IOU_THRESHOLD          = 0.5






