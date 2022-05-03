#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:14:19
#   Description :
#
#================================================================

import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    anchors = anchors.reshape(-1, 2)

    return anchors.reshape(3, 3, 2)


def image_preporcess(image1, image2,image3,target_sizex,target_sizey, gt_boxes=None):

    # image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # print(image1.shape)
    # # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # image1 = image1[:, :, np.newaxis]
    ih, iw    = target_sizex,target_sizey
    h,  w ,_= image1.shape
    
    # h,  w, _  = image2.shape
    #print(image2.shape)
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image1_resized = cv2.resize(image1, (nw, nh))
    # image1_resized = image1_resized[:, :, np.newaxis]
    image2_resized = cv2.resize(image2, (nw, nh))
    image3_resized = cv2.resize(image3, (nw, nh))
    # image4_resized = cv2.resize(image4, (nw, nh))
    # image5_resized = cv2.resize(image5, (nw, nh))
    # image2_resized = image2_resized[:, :, np.newaxis]
    # image3_resized = image3_resized[:, :, np.newaxis]
    # image4_resized = image4_resized[:, :, np.newaxis]
    # image5_resized = image5_resized[:, :, np.newaxis]

    image_paded1 = np.full(shape=[ih, iw, 3], fill_value=128.0)
    image_paded2 = np.full(shape=[ih, iw, 3], fill_value=128.0)
    image_paded3 = np.full(shape=[ih, iw, 3], fill_value=128.0)
    # image_paded4 = np.full(shape=[ih, iw, 1], fill_value=0.5)
    # image_paded5 = np.full(shape=[ih, iw, 1], fill_value=0.5)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded1[dh:nh+dh, dw:nw+dw, :] = image1_resized
    #print(image_paded1)
    image_paded2[dh:nh+dh, dw:nw+dw, :] = image2_resized
    image_paded3[dh:nh + dh, dw:nw + dw, :] = image3_resized
    # image_paded4[dh:nh + dh, dw:nw + dw, :] = image4_resized
    # image_paded5[dh:nh + dh, dw:nw + dw, :] = image5_resized
    image_paded1 = image_paded1 / 255.
    image_paded2 = image_paded2 / 255.
    image_paded3 = image_paded3 / 255.
    # image_paded4 = (np.log(image_paded4+1)*50.+12.5)/255.
    # image_paded5 = np.log(image_paded5+1)*50.+12.5
    if gt_boxes is None:
        return image_paded1,image_paded2,image_paded3
    #
    # else:
    #     gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
    #     gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
    #     return image_paded1,image_paded2,image_paded3, gt_boxes
    elif gt_boxes != []:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded1, image_paded2, image_paded3, gt_boxes

    else:
        return image_paded1, image_paded2, image_paded3,gt_boxes


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w,_= image.shape
    hsv_tuples = [(1.0 * x / 3, 1., 1.) for x in range(3)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 1
        # score = bbox[4]
        # class_ind = int(bbox[5])
        score = 1.0
        class_ind = 2
        bbox_color = colors[2]
        bbox_thick = int(0.8 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        #cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious



def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='soft-nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        R = True
        if R == True:

            w = np.ones((len(cls_bboxes), len(cls_bboxes)), dtype=np.float64)
            b = np.ones((len(cls_bboxes), len(cls_bboxes)), dtype=np.float64)
            # w = np.dot(bboxes[:, :4], np.transpose(bboxes[:, :4]))
            for i in range(len(cls_bboxes)):
                for j in range(len(cls_bboxes)):
                    w[i, j] = 1 / (1.001 - bboxes_giou(cls_bboxes[i, :4], cls_bboxes[j, :4]))
                    _, b[i, j] = bboxes_diou(cls_bboxes[i, :4], cls_bboxes[j, :4])
                    b[i, j] = 1 / (b[i, j] + 0.001)

            for i in range(len(cls_bboxes)):
                sum1 = 0
                sum2 = 0
                for j in range(len(cls_bboxes)):
                    sum1 = sum1 + w[i, j]
                    sum2 = sum2 + b[i, j]
                for k in range(len(cls_bboxes)):
                    w[i, k] = w[i, k] / sum1
                    b[i, k] = b[i, k] / sum2
            cls_bboxes[:, 4] = np.dot(b, cls_bboxes[:, 4])  # 中心点校正
            # bboxes[:, 4] = np.dot(w, bboxes[:, 4])#iou校正
            cls_bboxes[:, 4] = np.dot(b, cls_bboxes[:, 4])  # 中心点校正
            # bboxes[:, 4] = np.dot(b, bboxes[:, 4])  # 中心点校正

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.1
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_sizex,input_sizey, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    #print(pred_xywh)
    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    #print(pred_coor)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_sizey / org_w, input_sizex / org_h)

    dw = (input_sizey - resize_ratio * org_w) / 2
    dh = (input_sizex - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    #print(pred_coor)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    invalid_mask1 = np.logical_or(( (pred_coor[:, 2]-pred_coor[:, 0])>600 ), (( pred_coor[:, 3]-pred_coor[:, 1]) >250))
    pred_coor[invalid_mask1] = 0
    #print(pred_coor)
    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    # print(bboxes_scale)
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    #print(classes)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    #print(scores)
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
    
    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
def bboxes_diou(boxes1,boxes2):
    '''
    cal DIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    #cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    #cal Intersection
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)

    # cal outer boxes
    outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[..., 0]) + np.square(outer[..., 1])

    # cal center distance
    boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dis = np.square(boxes1_center[..., 0] - boxes2_center[..., 0]) + \
                 np.square(boxes1_center[..., 1] - boxes2_center[..., 1])

    # cal diou
    dious = ious - center_dis / outer_diagonal_line
    dig = center_dis
    return dious,dig

def bboxes_giou(boxes1,boxes2):
    '''
    cal GIOU of two boxes or batch boxes
    such as: (1)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[15,15,25,25]])
            boxes2 = np.asarray([[5,5,10,10]])
            and res is [-0.49999988  0.25       -0.68749988]
            (2)
            boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            boxes2 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
            and res is [1. 1. 1.]
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # ===========cal IOU=============#
    #cal Intersection
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area

    return gious



