#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-03-15 18:05:03
#   Description :
#
#================================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
import scipy.io as scio




class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizesx = cfg.TRAIN.INPUT_SIZEX if dataset_type == 'train' else cfg.TEST.INPUT_SIZEX
        self.input_sizesy = cfg.TRAIN.INPUT_SIZEY if dataset_type == 'train' else cfg.TEST.INPUT_SIZEY
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizesx = self.input_sizesx
        self.train_input_sizesy = cfg.TRAIN.INPUT_SIZEY
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)

        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[0:]) != 0]
            # annotations = [line.strip() for line in txt]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_input_sizex = random.choice(self.train_input_sizesx)
            self.train_input_sizey = self.train_input_sizex*3
            self.train_output_sizesx = self.train_input_sizex // self.strides
            self.train_output_sizesy = self.train_input_sizey // self.strides
            self.anchors = np.array(
                utils.get_anchors(cfg.YOLO.ANCHORS))

            batch_image1 = np.zeros((self.batch_size, self.train_input_sizex, self.train_input_sizey, 3))
            batch_image2 = np.zeros((self.batch_size, self.train_input_sizex, self.train_input_sizey, 1))
            batch_image3 = np.zeros((self.batch_size, self.train_input_sizex, self.train_input_sizey, 1))
            batch_image4 = np.zeros((self.batch_size, self.train_input_sizex, self.train_input_sizey, 1))
            batch_image5 = np.zeros((self.batch_size, self.train_input_sizex, self.train_input_sizey, 1))
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizesx[0], self.train_output_sizesy[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizesx[1], self.train_output_sizesy[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizesx[2], self.train_output_sizesy[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image1, image2,image3,image4,image5,bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image1[num, :, :, :] = image1
                    #print(image1)
                    batch_image2[num, :, :, :] = image2
                    batch_image3[num, :, :, :] = image3
                    batch_image4[num, :, :, :] = image4
                    batch_image5[num, :, :, :] = image5
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image1, batch_image2,batch_image3,batch_image4,batch_image5,batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image1,image2,image3,image4,image5, bboxes):

        if random.random() < 0.7:
            _, w, _ = image1.shape
            image1 = image1[:, ::-1, :]
            image2 = image2[:, ::-1, :]
            image3 = image3[:, ::-1, :]
            image4 = image4[:, ::-1, :]
            image5 = image5[:, ::-1, :]
            if bboxes != []:
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]


        return image1,image2,image3,image4,image5, bboxes

    def random_crop(self, image1, image2,image3,image4,image5,bboxes):

        if random.random() < 0.7:
            if bboxes != []:
                h, w, _ = image1.shape
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

                max_l_trans = max_bbox[0]
                max_u_trans = max_bbox[1]
                max_r_trans = w - max_bbox[2]
                max_d_trans = h - max_bbox[3]

                crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
                crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
                crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
                crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

                image1 = image1[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
                image2 = image2[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
                image3 = image3[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
                image4 = image4[crop_ymin: crop_ymax, crop_xmin: crop_xmax]
                image5 = image5[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image1,image2,image3,image4,image5,  bboxes

    def random_translate(self, image1,image2,image3,image4,image5,  bboxes):

        if random.random() < 0.7:
            if bboxes != []:
                h, w, _ = image1.shape
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

                max_l_trans = max_bbox[0]
                max_u_trans = max_bbox[1]
                max_r_trans = w - max_bbox[2]
                max_d_trans = h - max_bbox[3]

                tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
                ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

                M = np.array([[1, 0, tx], [0, 1, ty]])
                image1 = cv2.warpAffine(image1, M, (w, h))
                image2 = cv2.warpAffine(image2, M, (w, h))
                image3 = cv2.warpAffine(image3, M, (w, h))
                image4 = cv2.warpAffine(image4, M, (w, h))
                image5 = cv2.warpAffine(image5, M, (w, h))

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image1, image2,image3,image4,image5,bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path1 = line[0]
        image_path2=line[1]
        image_path3 = line[2]
        image_path4 = line[3]
        image_path5 = line[4]
        if not os.path.exists(image_path1):
            raise KeyError("%s does not exist ... " %image_path1)
        if not os.path.exists(image_path2):
            raise KeyError("%s does not exist ... " %image_path2)
        if not os.path.exists(image_path3):
            raise KeyError("%s does not exist ... " %image_path3)
        image1 = np.array(cv2.imread(image_path1))#数组形式
        # image1 = image1[:, :, np.newaxis]
        #print(image1)
        #print(image1.shape)
        image2_data = scio.loadmat(image_path2)
        image2 = image2_data['depth']
        image3_data = scio.loadmat(image_path3)
        image3 = image3_data['depth']
        image4_data = scio.loadmat(image_path4)
        image4 = image4_data['depth']
        image5_data = scio.loadmat(image_path5)
        image5 = image5_data['depth']
        image2 = image2[:, :, np.newaxis]
        image3 = image3[:, :, np.newaxis]
        image4 = image4[:, :, np.newaxis]
        image5 = image5[:, :, np.newaxis]
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[5:]])
        if self.data_aug:
            image1,image2,image3,image4,image5, bboxes = self.random_horizontal_flip(np.copy(image1),np.copy(image2),np.copy(image3), np.copy(image4),np.copy(image5),np.copy(bboxes))
            image1,image2,image3,image4,image5, bboxes = self.random_crop(np.copy(image1),np.copy(image2),np.copy(image3), np.copy(image4),np.copy(image5),np.copy(bboxes))
            image1,image2,image3,image4,image5, bboxes = self.random_translate(np.copy(image1),np.copy(image2),np.copy(image3),np.copy(image4),np.copy(image5), np.copy(bboxes))

        #print(bboxes)
        image1,image2, image3,image4, image5,bboxes = utils.image_preporcess(np.copy(image1), np.copy(image2),np.copy(image3),np.copy(image4),np.copy(image5),self.train_input_sizex, self.train_input_sizey, np.copy(bboxes))

        updated_bb = []
        for bb in bboxes:
            x1, y1, x2, y2, cls_label = bb
            
            if x2 <= x1 or y2 <= y1:
                # dont use such boxes as this may cause nan loss.
                continue

            x1 = int(np.clip(x1, 0, image1.shape[1]))
            y1 = int(np.clip(y1, 0, image1.shape[0]))
            x2 = int(np.clip(x2, 0, image1.shape[1]))
            y2 = int(np.clip(y2, 0, image1.shape[0]))
            # clipping coordinates between 0 to image dimensions as negative values 
            # or values greater than image dimensions may cause nan loss.
            updated_bb.append([x1, y1, x2, y2, cls_label])

        return image1,image2,image3,image4,image5, np.array(updated_bb)

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / (union_area + 1e-6)
        # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss


    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizesx[i], self.train_output_sizesy[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]*self.train_output_sizesx[i]*3

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    xind = np.clip(xind, 0, self.train_output_sizesy[i] - 1)
                    yind = np.clip(yind, 0, self.train_output_sizesx[i] - 1)
                    # This will mitigate errors generated when the location computed by this is more the grid cell location. 
                    # e.g. For 52x52 grid cells possible values of xind and yind are in range [0-51] including both. 
                    # But sometimes the coomputation makes it 52 and then it will try to find that location in label array 
                    # which is not present and throws error during training.

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                xind = np.clip(xind, 0, self.train_output_sizesy[i] - 1)
                yind = np.clip(yind, 0, self.train_output_sizesx[i] - 1)
                # This will mitigate errors generated when the location computed by this is more the grid cell location. 
                # e.g. For 52x52 grid cells possible values of xind and yind are in range [0-51] including both. 
                # But sometimes the coomputation makes it 52 and then it will try to find that location in label array 
                # which is not present and throws error during training.

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs




