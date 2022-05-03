# -*- coding: utf-8 -*-
"""
Created on Sat May 29 08:42:06 2021

@author: WIN
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:35:04 2021

@author: WIN
"""

#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from crossfusiongan import YOLOV3
from crossGenerator import Generator
image_width = 768
image_height = 256



def main():
    #input_size       = cfg.TEST.INPUT_SIZE
    anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
    classes          = utils.read_class_names(cfg.YOLO.CLASSES)
    num_classes      = len(classes)
    anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
    score_threshold  = cfg.TEST.SCORE_THRESHOLD
    iou_threshold    = cfg.TEST.IOU_THRESHOLD
    moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
    annotation_path  = cfg.TEST.ANNOT_PATH
    weight_file      = cfg.TEST.WEIGHT_FILE
    write_image      = cfg.TEST.WRITE_IMAGE
    write_image_path = cfg.TEST.WRITE_IMAGE_PATH
    show_label       = cfg.TEST.SHOW_LABEL
    input_sizex       = cfg.TEST.INPUT_SIZEX
    input_sizey       = cfg.TEST.INPUT_SIZEY
    
    image_vi = tf.compat.v1.placeholder(tf.float32, [1, image_height, image_width, 3], name='image_vi')
    image_de = tf.compat.v1.placeholder(tf.float32, [1, image_height, image_width, 3], name='image_de')
    image_in = tf.compat.v1.placeholder(tf.float32, [1, image_height, image_width, 3], name='image_in')
    trainable     = tf.placeholder(dtype=tf.bool, name='trainable')
    #input_image=tf.concat([image_ir, image_vi], axis=-1)
    G = Generator('Generator')
    fusion_image = G.transform(image_vi, image_de, image_in)
    #fusion_image= generator(image_vi,image_de,image_in,reuse=False)  # 得到生成的y域图像
    #print(fusion_image.shape)
  
   
    model = YOLOV3(fusion_image,trainable)

    conv_lbbox, conv_mbbox, conv_sbbox=model.build_network(fusion_image)
    conv_lbbox = model.decode(conv_lbbox, anchors[2], 32)
    conv_mbbox = model.decode(conv_mbbox, anchors[1], 16)
    conv_sbbox = model.decode(conv_sbbox, anchors[0], 8)
    
    restore_var1  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')#取出Generator的参数  #
    y_vars = []
    for var in tf.trainable_variables():
        var_name = var.op.name
        # print(var_name)
        var_name_mess = str(var_name).split('/')
        if var_name_mess[0] in ['darknet', 'conv52', 'conv53', 'conv54', 'conv55', 'conv56', 'conv57', 'conv58',
                                'conv59', 'conv60', 'conv61', 'conv62', 'conv63', 'conv64', 'conv65', 'conv66',
                                'conv67', 'conv68', 'conv_lobj_branch', 'conv_mobj_branch', 'conv_sobj_branch',
                                'conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
            y_vars.append(var)
    print(y_vars)
    # restore_var2 = [v for v in tf.global_variables() if 'D-yolo' in v.name]  # 需要载入的已训练的模型参数
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 设定显存不超量使用
    sess = tf.Session(config=config)  # 建立会话层
    sess.run(tf.global_variables_initializer())
    # saver1 = tf.train.Saver(var_list=restore_var1, max_to_keep=50)  # 导入模型参数时使用
    # saver2 = tf.train.Saver(var_list=y_vars, max_to_keep=50)  # 导入模型参数时使用
    # #checkpoint = tf.train.latest_checkpoint(self.weight_file)  # 读取模型参数
    # saver1.restore(sess, weight_file)  # 导入模型参数
    # saver2.restore(sess, weight_file)  # 导入模型参数
    # sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    #     # # self.saver = tf.train.Saver(ema_obj.variables_to_restore())
    # saver = tf.train.Saver()
    # saver.restore(sess, weight_file)
    ema_obj = tf.train.ExponentialMovingAverage(0.9)
    saver = tf.train.Saver(ema_obj.variables_to_restore())
    saver.restore(sess, weight_file)
    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(write_image_path): shutil.rmtree(write_image_path)
    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(write_image_path)

    with open(annotation_path, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path1 = annotation[0]
            image_path2 = annotation[1]
            image_path3 = annotation[2]
            image_name = image_path1.split('/')[-1]
            image1 = cv2.imread(image_path1)
            #image1 = image1[:, :, np.newaxis]
            image2 = cv2.imread(image_path2)
            image3 = cv2.imread(image_path3)
            # image2 = image2[:, :, np.newaxis]
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[3:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt=[]
                classes_gt=[]
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = classes[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            image_data1,image_data2,image_data3 = utils.image_preporcess(image1, image2, image3,input_sizex, input_sizey)
            IMG=image_data1
            image_data1 = image_data1[np.newaxis, ...]
            image_data2 = image_data2[np.newaxis, ...]
            image_data3 = image_data3[np.newaxis, ...]
            org_image = np.copy(image1)
            org_h, org_w, _ = org_image.shape
            feed_dict = {image_vi: image_data1, image_de: image_data2,image_in: image_data3,trainable:True}  # 建立feed_dict
            fusion_img,pred_lbbox, pred_mbbox, pred_sbbox= sess.run([fusion_image,conv_lbbox, conv_mbbox, conv_sbbox], feed_dict=feed_dict)  # 得到生成的y域图像与x域图像
            fusion_img=fusion_img[0,:,:,:]
            resize_ratio=min(768/org_w,256/org_h)
            fusion_img=cv2.resize(fusion_img,(int(768/resize_ratio),int(256/resize_ratio)))
            dw,dh=(org_w-int(768/resize_ratio))//2,(int(256/resize_ratio)-org_h)//2
            fusion_img=fusion_img[dh:dh+org_h,dw:dw+org_w,:]
            # print(fusion_img.shape)
            #print(pred_sbbox)
            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
            #print(pred_bbox)
            bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), input_sizex, input_sizey, score_threshold)
            #print(bboxes)
            bboxes = utils.nms(bboxes, iou_threshold)
            
            #print(write_image)
            if write_image:
                image = utils.draw_bbox(image1, bboxes, show_label=False)
                #print(image.shape)
                cv2.imwrite(write_image_path+image_name, image)
                # cv2.imwrite(write_image_path + image_name, fusion_img*255.)
                #print(image_name)

            with open(predict_result_path, 'w') as f:
                for bbox in bboxes:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = classes[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
if __name__ == '__main__':
    main()

