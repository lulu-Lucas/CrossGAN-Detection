import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.05


class Generator(object):

    def __init__(self, sco):
        self.encoder = Encoder(sco)
        self.decoder = Decoder(sco)

    def transform(self, images_vi, images_de,images_in):
        # img = tf.concat([images_vi, images_de,images_in], 3)
        code = self.encoder.encode(images_vi, images_de,images_in)
        self.target_features = code
        #print(code.shape)
        generated_img = self.decoder.decode(self.target_features)
        return generated_img


class Encoder(object):
    def __init__(self, scope_name):
        self.scope = scope_name
        self.weight_vars = []
        with tf.variable_scope(self.scope):
            with tf.variable_scope('encoder'):
                self.weight_vars.append(self._create_variables(3, 16, 3, scope = 'conv1_1'))
                self.weight_vars.append(self._create_variables(48, 16, 3, scope = 'dense_block_conv1'))
                self.weight_vars.append(self._create_variables(96, 16, 3, scope = 'dense_block_conv2'))
                self.weight_vars.append(self._create_variables(144, 16, 3, scope = 'dense_block_conv3'))
                self.weight_vars.append(self._create_variables(192, 16, 3, scope = 'dense_block_conv4'))
        self.weight_vars2 = []
        with tf.variable_scope(self.scope):
            with tf.variable_scope('encoder'):
                self.weight_vars2.append(self._create_variables(3, 16, 3, scope='conv2_1'))
                self.weight_vars2.append(self._create_variables(48, 16, 3, scope='conv2_2'))
                self.weight_vars2.append(self._create_variables(96, 16, 3, scope='conv2_3'))
                self.weight_vars2.append(self._create_variables(144, 16, 3, scope='conv2_4'))
                self.weight_vars2.append(self._create_variables(192, 16, 3, scope='conv2_5'))
        self.weight_vars3 = []
        with tf.variable_scope(self.scope):
            with tf.variable_scope('encoder'):
                self.weight_vars3.append(self._create_variables(3, 16, 3, scope='conv3_1'))
                self.weight_vars3.append(self._create_variables(48, 16, 3, scope='conv3_2'))
                self.weight_vars3.append(self._create_variables(96, 16, 3, scope='conv3_3'))
                self.weight_vars3.append(self._create_variables(144, 16, 3, scope='conv3_4'))
                self.weight_vars3.append(self._create_variables(192, 16, 3, scope='conv3_5'))
        self.cross_vars1 = []
        with tf.variable_scope(self.scope):
            with tf.variable_scope('encoder'):
                self.cross_vars1.append(self._create_variables(16, 16, 1, scope='cross1_1'))
                self.cross_vars1.append(self._create_variables(16, 16, 1, scope='cross1_2'))
                self.cross_vars1.append(self._create_variables(16, 16, 1, scope='cross1_3'))
                self.cross_vars1.append(self._create_variables(16, 16, 1, scope='cross1_4'))
                self.cross_vars1.append(self._create_variables(16, 16, 1, scope='cross1_5'))
        self.cross_vars2 = []
        with tf.variable_scope(self.scope):
            with tf.variable_scope('encoder'):
                self.cross_vars2.append(self._create_variables(16, 16, 1, scope='cross2_1'))
                self.cross_vars2.append(self._create_variables(16, 16, 1, scope='cross2_2'))
                self.cross_vars2.append(self._create_variables(16, 16, 1, scope='cross2_3'))
                self.cross_vars2.append(self._create_variables(16, 16, 1, scope='cross2_4'))
                self.cross_vars2.append(self._create_variables(16, 16, 1, scope='cross2_5'))
        self.cross_vars3 = []
        with tf.variable_scope(self.scope):
            with tf.variable_scope('encoder'):
                self.cross_vars3.append(self._create_variables(16, 16, 1, scope='cross3_1'))
                self.cross_vars3.append(self._create_variables(16, 16, 1, scope='cross3_2'))
                self.cross_vars3.append(self._create_variables(16, 16, 1, scope='cross3_3'))
                self.cross_vars3.append(self._create_variables(16, 16, 1, scope='cross3_4'))
                self.cross_vars3.append(self._create_variables(16, 16, 1, scope='cross3_5'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        shape = [kernel_size, kernel_size, input_filters, output_filters]
        with tf.variable_scope(scope):
            kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV),
                                 name = 'kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
        return (kernel, bias)

    def encode(self, image1,image2,image3):
        dense_indices = [1, 2, 3, 4, 5]
        #
        out1 = image1
        out2 = image2
        out3 = image3
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            kernel2, bias2 = self.weight_vars2[i]
            kernel3, bias3 = self.weight_vars3[i]
            crosskernel1,crossbias1=self.cross_vars1[i]
            crosskernel2, crossbias2 = self.cross_vars2[i]
            crosskernel3, crossbias3 = self.cross_vars3[i]
            if i in dense_indices:
                out0_1,out1 = conv2d(out1, kernel, bias, dense = True, use_relu = True,
                             Scope = self.scope + '/encoder/b1' + str(i))
                out0_1=convcross(out0_1,crosskernel1, crossbias1,Scope=self.scope + '/encoder/cross1' + str(i),BN = False)
                out0_2,out2 = conv2d(out2, kernel2, bias2, dense=True, use_relu=True,
                             Scope=self.scope + '/encoder/b2' + str(i))
                out0_2 = convcross(out0_2, crosskernel2, crossbias2, Scope=self.scope + '/encoder/cross2' + str(i),BN = False)
                out0_3,out3 = conv2d(out3, kernel3, bias3, dense=True, use_relu=True,
                             Scope=self.scope + '/encoder/b3' + str(i))
                out0_3 = convcross(out0_3, crosskernel3, crossbias3, Scope=self.scope + '/encoder/cross3' + str(i),BN = False)
            else:
                out0_1,out1 = conv2d(out1, kernel, bias, dense=False, use_relu=True,
                              Scope=self.scope + '/encoder/b1' + str(i))
                out0_1 = convcross(out0_1, crosskernel1, crossbias1, Scope=self.scope + '/encoder/cross1' + str(i),BN = False)
                out0_2,out2 = conv2d(out2, kernel2, bias2, dense=False, use_relu=True,
                              Scope=self.scope + '/encoder/b2' + str(i))
                out0_2 = convcross(out0_2, crosskernel2, crossbias2, Scope=self.scope + '/encoder/cross2' + str(i),BN = False)
                out0_3,out3 = conv2d(out3, kernel3, bias3, dense=False, use_relu=True,
                              Scope=self.scope + '/encoder/b3' + str(i))
                out0_3 = convcross(out0_3, crosskernel3, crossbias3, Scope=self.scope + '/encoder/cross3' + str(i),BN = False)
            out1=tf.concat([out1, out0_2,out0_3], 3)
            #print(out1.shape)
            out2 = tf.concat([out0_1, out2, out0_3], 3)
            out3 = tf.concat([out0_1, out0_2, out3], 3)
        # out=tf.concat([out1, out2, out3], 3)
        #print(out.shape)

        return out1


class Decoder(object):
    def __init__(self, scope_name):
        self.weight_vars = []
        self.scope = scope_name
        with tf.name_scope(scope_name):
            with tf.variable_scope('decoder'):
                self.weight_vars.append(self._create_variables(240, 240, 3, scope = 'conv2_1'))
                self.weight_vars.append(self._create_variables(240, 128, 3, scope = 'conv2_2'))
                self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_3'))
                self.weight_vars.append(self._create_variables(64, 32, 3, scope = 'conv2_4'))
                self.weight_vars.append(self._create_variables(32, 3, 3, scope = 'conv2_5'))


    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
            bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
        return (kernel, bias)

    def decode(self, image):
        final_layer_idx = len(self.weight_vars) - 1

        out = image
        print(out.shape)
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            if i == 0:
                out0,out = conv2d(out, kernel, bias, dense = False, use_relu = True,
                             Scope = self.scope + '/decoder/b' + str(i), BN = False)
            if i == final_layer_idx:
                out0,out = conv2d(out, kernel, bias, dense = False, use_relu = False,
                             Scope = self.scope + '/decoder/b' + str(i), BN = False)
                out = tf.nn.tanh(out) / 2 + 0.5
            if (i==1)or(i==2)or(i==3):
                out0,out = conv2d(out, kernel, bias, dense = False, use_relu = True, BN = True,
                             Scope = self.scope + '/decoder/b' + str(i))
        return out


def conv2d(x, kernel, bias, dense = False, use_relu = True, Scope = None, BN = True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides = [1, 1, 1, 1], padding = 'VALID')
    out = tf.nn.bias_add(out, bias)
    out0=out
    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training = True)
    if use_relu:
        out = tf.nn.relu(out)
    if dense:
        # out0 = out
        out = tf.concat([out, x], 3)

    return out0,out
def convcross(x, kernel, bias,Scope = None,BN = True):
    # padding image with reflection mode
    # x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
    # conv and add bias
    out = tf.nn.conv2d(x, kernel, strides = [1, 1, 1, 1], padding = 'VALID')
    out = tf.nn.bias_add(out, bias)
    if BN:
        with tf.variable_scope(Scope):
            out = tf.layers.batch_normalization(out, training = True)

    return out

def up_sample(x, scale_factor = 2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size = new_size)