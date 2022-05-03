import tensorflow as tf
import numpy as np


def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a different scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    value = tf.reduce_mean(value)
    return value


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def L1_LOSS(batchimg):
    L1_norm = tf.reduce_sum(tf.abs(batchimg), axis=[1, 2])
    # tf.norm(batchimg, axis = [1, 2], ord = 1) / int(batchimg.shape[1])
    E = tf.reduce_mean(L1_norm)
    return E


def Fro_LOSS(batchimg):
    fro_norm = tf.square(tf.norm(batchimg, axis=[1, 2], ord='fro'))
    # / (int(batchimg.shape[1]) * int(batchimg.shape[2]))
    E = tf.reduce_mean(fro_norm)
    return E


def discriminator_loss(Ra, loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if Ra and loss_func.__contains__('wgan'):
        print("No exist [Ra + WGAN], so use the {} loss function".format(loss_func))
        Ra = False

    if Ra:
        real_logit = (real - tf.reduce_mean(fake))
        fake_logit = (fake - tf.reduce_mean(real))

        if loss_func == 'lsgan':
            real_loss = tf.reduce_mean(tf.square(real_logit - 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake_logit + 1.0))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan':
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real_logit))
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake_logit))

        if loss_func == 'hinge':
            real_loss = tf.reduce_mean(relu(1.0 - real_logit))
            fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))

    else:
        if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
            real_loss = -tf.reduce_mean(real)
            fake_loss = tf.reduce_mean(fake)

        if loss_func == 'lsgan':
            real_loss = tf.reduce_mean(tf.square(real - 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan':
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

        if loss_func == 'hinge':
            real_loss = tf.reduce_mean(relu(1.0 - real))
            fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss


def relu(x):
    return tf.nn.relu(x)
def generator_loss(Ra, loss_func, real, fake):
    fake_loss = 0
    real_loss = 0

    if Ra and loss_func.__contains__('wgan') :
        print("No exist [Ra + WGAN], so use the {} loss function".format(loss_func))
        Ra = False

    if Ra :
        fake_logit = (fake - tf.reduce_mean(real))
        real_logit = (real - tf.reduce_mean(fake))

        if loss_func == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.square(fake_logit - 1.0))
            real_loss = tf.reduce_mean(tf.square(real_logit + 1.0))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake_logit))
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real_logit))

        if loss_func == 'hinge' :
            fake_loss = tf.reduce_mean(relu(1.0 - fake_logit))
            real_loss = tf.reduce_mean(relu(1.0 + real_logit))

    else :
        if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
            fake_loss = -tf.reduce_mean(fake)

        if loss_func == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

        if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

        if loss_func == 'hinge' :
            fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss + real_loss

    return loss
