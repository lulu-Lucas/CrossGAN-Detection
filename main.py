# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:00:20 2021

@author: WIN
"""

import os
import cv2
import tensorflow as tf

import core.common as common
import core.utils as utils
import core.backbone as backbone
from core.config import cfg
import numpy as np
from core.dataset import Dataset
from tqdm import tqdm
from skimage import transform,data
from crossfusiongan import *
import tensorflow.contrib.slim as slim
import math
from crossGenerator import Generator
from Discriminator import Discriminator1, Discriminator2,Discriminator3
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss,discriminator_loss ,generator_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_width = 512
image_height = 512

epochs = 400
images_vi = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='images_vi')
labels_vi = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='labels_vi')

images_de = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='images_de')
labels_de = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='labels_de')

images_in = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='images_in')
labels_in = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='labels_in')


label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
trainable     = tf.placeholder(dtype=tf.bool, name='training')
lr = tf.placeholder(tf.float32, None, name='learning_rate')  # 训练中的学习率
lr0 = tf.placeholder(tf.float32, None, name='learning_rate0')  # 训练中的学习率
lr1 = tf.placeholder(tf.float32, None, name='learning_rate1')  # 训练中的学习率
#input_image=tf.concat([images_ir, images_vi], axis=-1)

G = Generator('Generator')
fusion_image = G.transform(images_vi,images_de,images_in)

D1 = Discriminator1('Discriminator1')
D2 = Discriminator2('Discriminator2')
D3 = Discriminator3('Discriminator3')

pos = D1.discrim(labels_vi, reuse = False)
pos2 = D2.discrim(labels_vi, reuse = False)
pos3= D3.discrim(labels_vi, reuse = False)
neg = D1.discrim(fusion_image, reuse=True)
neg2 = D2.discrim(fusion_image, reuse=True)
neg3 = D3.discrim(fusion_image, reuse=True)
# Loss for Generator
G_loss_GAN_D1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg))) # 生成器对抗损失（vi）
# G_loss_GAN_D1 = generator_loss(True, 'lsgan', real=pos, fake=neg)
G_loss_GAN_D2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg2, labels=tf.ones_like(neg2))) # 生成器对抗损失（de）
G_loss_GAN_D3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg3, labels=tf.ones_like(neg3))) # 生成器对抗损失（in）
#G_loss_GAN = G_loss_GAN_D1 + G_loss_GAN_D2+G_loss_GAN_D3
G_loss_GAN1 = G_loss_GAN_D1
G_loss_GAN2 = G_loss_GAN_D1+G_loss_GAN_D2
G_loss_GAN3 = G_loss_GAN_D1+G_loss_GAN_D2+G_loss_GAN_D3
LOSS_IN = Fro_LOSS(fusion_image - labels_in)
LOSS_VI = L1_LOSS(gradient(fusion_image) - gradient(labels_vi))
G_loss_norm = LOSS_IN / 48000000 + 0.000000004 * LOSS_VI # 传统损失函数
# G_loss_norm =  1.2 * LOSS_VI
G_loss1 = G_loss_GAN1*5 + 0.6 * G_loss_norm
G_loss2 = G_loss_GAN2*5 + 0.6 * G_loss_norm
G_loss3 = G_loss_GAN3*5 + 0.6 * G_loss_norm
# G_loss = 0.6 * G_loss_norm
# Loss for Discriminator1
D1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos, labels=tf.ones_like(pos)))
D1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.zeros_like(neg)))
D1_loss = D1_loss_fake + D1_loss_real
# D1_loss = discriminator_loss(True, 'lsgan', real=pos, fake=neg)
D1_loss = D1_loss*5
# Loss for Discriminator2
D2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos2, labels=tf.ones_like(pos2)))
D2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg2, labels=tf.zeros_like(neg2)))
D2_loss = D2_loss_fake + D2_loss_real
D2_loss = D2_loss*5
# # Loss for Discriminator3
D3_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pos3, labels=tf.ones_like(pos3)))
D3_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg3, labels=tf.zeros_like(neg3)))
D3_loss = D3_loss_fake + D3_loss_real
D3_loss = D3_loss*5

#检测模型
model = YOLOV3(fusion_image, trainable)
giou_loss, conf_loss, prob_loss = model.compute_loss(
                                                    label_sbbox,  label_mbbox,  label_lbbox,
                                                    true_sbboxes, true_mbboxes, true_lbboxes)
y_loss=giou_loss + conf_loss + prob_loss

g_loss1=G_loss1+y_loss
g_loss2=G_loss2+y_loss
g_loss3=G_loss3+y_loss

g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')#取出Generator的参数
d1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator1')
d2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator2')
d3_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator3')

y_vars= []
for var in tf.global_variables():
    var_name = var.op.name
    #print(var_name)
    var_name_mess = str(var_name).split('/')
    if var_name_mess[0] in ['darknet','conv52','conv53','conv54','conv55','conv56','conv57','conv58','conv59','conv60','conv61','conv62','conv63','conv64','conv65','conv66','conv67','conv68','conv_lobj_branch','conv_mobj_branch','conv_sobj_branch','conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
        y_vars.append(var)

first_stage_trainable_var_list= []
for var in tf.trainable_variables():
    var_name = var.op.name
    #print(var_name)
    var_name_mess = str(var_name).split('/')
    if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
        first_stage_trainable_var_list.append(var)


# train_g1 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(g_loss1, var_list=g_vars)
#train_g2 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(g_loss2, var_list=g_vars)
train_g3 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(g_loss3, var_list=g_vars)
train_d1 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(D1_loss, var_list=d1_vars)
train_d2 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(D2_loss, var_list=d2_vars)
train_d3 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(D3_loss, var_list=d3_vars)
train_y = tf.train.AdamOptimizer(learning_rate=lr1).minimize(y_loss, var_list=y_vars)
train_y0 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(y_loss, var_list=first_stage_trainable_var_list)
moving_ave = tf.train.ExponentialMovingAverage(0.9).apply(tf.trainable_variables())
vars1= []
for var in tf.global_variables():
    var_name = var.op.name
    #print(var_name)
    var_name_mess = str(var_name).split('/')
    if var_name_mess[0] in ['darknet','conv52','conv53','conv54','conv55','conv56','conv57','conv58','conv59','conv60','conv61','conv62','conv63','conv64','conv65','conv66','conv67','conv68','conv_lobj_branch','conv_mobj_branch','conv_sobj_branch','conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
        vars1.append(var)
vars2= []
for var in tf.global_variables():
    var_name = var.op.name
    #print(var_name)
    var_name_mess = str(var_name).split('/')
    if var_name_mess[0] in ['Generator','Generator_1','Generator_2','Generator_3','Generator_4','Generator_5','Generator_6','Generator_7']:
        vars2.append(var)
vars3= []
for var in tf.global_variables():
    var_name = var.op.name
    #print(var_name)
    var_name_mess = str(var_name).split('/')
    if var_name_mess[0] in ['Discriminator1']:
        vars3.append(var)
vars4= []
for var in tf.global_variables():
    var_name = var.op.name
    #print(var_name)
    var_name_mess = str(var_name).split('/')
    if var_name_mess[0] in ['Discriminator2']:
        vars4.append(var)
vars5= []
for var in tf.global_variables():
    var_name = var.op.name
    #print(var_name)
    var_name_mess = str(var_name).split('/')
    if var_name_mess[0] in ['Discriminator3']:
        vars5.append(var)
load_vars = vars1 + vars2 + vars3
vars = vars1 + vars2 + vars3 + vars4 + vars5
print(load_vars)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    with tf.control_dependencies([train_y0]):
        with tf.control_dependencies([moving_ave]):
            train_y0 = tf.no_op()
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    with tf.control_dependencies([train_y]):
        with tf.control_dependencies([moving_ave]):
            train_y = tf.no_op()
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    with tf.control_dependencies([train_d1]):
        with tf.control_dependencies([moving_ave]):
            train_d1 = tf.no_op()
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#     with tf.control_dependencies([train_g1]):
#         with tf.control_dependencies([moving_ave]):
#             train_g1 = tf.no_op()
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#     with tf.control_dependencies([train_g2]):
#         with tf.control_dependencies([moving_ave]):
#             train_g2 = tf.no_op()
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    with tf.control_dependencies([train_g3]):
        with tf.control_dependencies([moving_ave]):
            train_g3 = tf.no_op()
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    with tf.control_dependencies([train_d2]):
        with tf.control_dependencies([moving_ave]):
            train_d2 = tf.no_op()
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    with tf.control_dependencies([train_d3]):
        with tf.control_dependencies([moving_ave]):
            train_d3 = tf.no_op()
trainset            = Dataset('train')
# testset            = Dataset('test')

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())#初始化模型参数


    # loader1 = tf.train.Saver(var_list=y_vars)  # 模型保存器
    # loader1.restore(sess,'kitti/pre')
    # loader2 = tf.train.Saver(var_list=load_vars)  # 模型保存器
    # loader2.restore(sess, 'cross/cross/model-124000')
    saver = tf.train.Saver(var_list=vars,max_to_keep=50)  # 模型保存器
    saver.restore(sess, 'cross/model-164000')


    step=164000
    for epoch in range(epochs):
        pbar = tqdm(trainset)
        for train_data in pbar:

            #忽略
            if (step >= 0) and (step <= 100000):
                lrate = 0.000001 + 0.5 * (0.0001 - 0.000001) * (
                            1 + math.cos((step) / (100000 ) * np.pi))  # 得到该训练epoch的学习率
            if step > 100000:
                lrate = 0.000001
            #学习率在下面
            if step<=10000:
                lrate1 = 0.0001*(step+1000)/(10000+1000)
            if (step>10000)and(step <= 204000):
                lrate1=0.000001 + 0.5 * (0.0001 - 0.000001) *(1 + math.cos((step - 10000) / (204000 - 10000) * np.pi))# 得到该训练epoch的学习率
            if step>204000:
                lrate1=0.000001
            #喂值
            feed_dict = {lr: lrate,
                         lr1: lrate1,
                         images_vi: train_data[0], images_de: train_data[1], images_in: train_data[2],
                         labels_vi: train_data[0], labels_de: train_data[1], labels_in: train_data[2],
                         label_sbbox: train_data[3],
                         label_mbbox: train_data[4],
                         label_lbbox: train_data[5],
                         true_sbboxes: train_data[6],
                         true_mbboxes: train_data[7],
                         true_lbboxes: train_data[8],
                         trainable: True}

            if step<124000:

                _, _, _, gl, dl1, yl = sess.run([train_g1, train_y, train_d1, g_loss1, D1_loss, y_loss],
                                                feed_dict=feed_dict)

            if (step>=124000)and(step<164000):
                _,_,_,_, gl, dl1,dl2,  yl = sess.run([train_g2, train_y,train_d1, train_d2,g_loss2, D1_loss,D2_loss, y_loss],
                                                       feed_dict=feed_dict)

            if step >= 164000:
                _, _, _, _,_, gl, dl1, dl2,dl3, yl = sess.run(
                    [train_g3, train_y, train_d1, train_d2, train_d3, g_loss3, D1_loss, D2_loss,D3_loss, y_loss],
                    feed_dict=feed_dict)


            step+=1
            print('epoch: %d, step: %d,learnrate: %f, Generator Loss: %f, DisLoss: %f,YOLO Loss: %f' % (epoch, step, lrate1,gl,dl3,yl))
            if (step%100)==0:
                #print(train_data[0])
                g = sess.run(fusion_image, feed_dict = {images_vi: train_data[0], images_de: train_data[1],images_in: train_data[2]})
                
                for j in range(1):
                    cv2.imwrite('crossresult/' + str(step) + "_"  + str(j)+'.png', g[j]*255.)
            if (step%4000)==0:
                # tbar = tqdm(testset)
                # loss = 0
                # yloss = 0
                # # saver.save(sess, 'kitti/model', step)
                # # saver1.save(sess, 'kitti/modelone', step)
                # #
                # # saver1.restore(sess, 'kitti/'+'modelone'+"-"+str(step))
                # #
                # # saver3 = tf.train.Saver(tf.trainable_variables())
                # # saver3.save(sess, 'kitti/model', step)
                # for test_data in tbar:
                #     gl, dl1,  yl = sess.run([g_loss, D1_loss, y_loss],
                #                           feed_dict={
                #                                         images_vi: test_data[0], images_de: test_data[1],
                #                                         images_in: test_data[2],
                #                                         labels_vi: test_data[0], labels_de: test_data[1],
                #                                         labels_in: test_data[2],
                #                                         label_sbbox: test_data[3],
                #                                         label_mbbox: test_data[4],
                #                                         label_lbbox: test_data[5],
                #                                         true_sbboxes: test_data[6],
                #                                         true_mbboxes: test_data[7],
                #                                         true_lbboxes: test_data[8],
                #                                         trainable: True})
                #     loss = loss + gl + dl1
                #     yloss = yloss + yl
                #     print('Loss: %f %f %d' % (gl + dl1 , yl, step))
                # loss = loss / 472.
                # yloss = yloss / 472.
                # saver1.save(sess, 'cross/modelone', step)
                #
                # saver1.restore(sess, 'cross/'+'modelone'+"-"+str(step))
                #
                # saver3 = tf.train.Saver(tf.trainable_variables())
                # saver3.save(sess, 'cross/model', step)
                saver.save(sess, 'cross/model',step)

