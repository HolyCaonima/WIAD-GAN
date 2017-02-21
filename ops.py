import numpy as np
import tensorflow as tf

import tensorflow.contrib.layers as ly

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def minibatch_disc(input_tensor, num_kernels=100, dim_per_kernel=5):
    batch_size = int(input_tensor.get_shape()[0])
    num_features = int(input_tensor.get_shape()[1])
    W = tf.Variable(tf.truncated_normal([num_features, num_kernels*dim_per_kernel], 0, 0.02))
    b = tf.Variable(tf.zeros([num_kernels]))
    activation = tf.matmul(input_tensor, W)
    activation = tf.reshape(activation, [batch_size, num_kernels, dim_per_kernel])
    tmp1 = tf.expand_dims(activation, 3)
    tmp2 = tf.transpose(activation, perm=[1,2,0])
    tmp2 = tf.expand_dims(tmp2, 0)
    abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
    f = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])
    f = f + b
    f = tf.concat(1, [input_tensor, f])
    return f

def resize_conv2d(input_tensor, num_filters, kernel_size, img_size,
                  act_fn=tf.nn.relu, norm_fn=ly.batch_norm,
                  weights_init=tf.random_normal_initializer(0, 0.02)):

    net = tf.image.resize_bilinear(input_tensor, img_size)
    net = ly.conv2d(net, num_filters, kernel_size, stride=1,
                    activation_fn=act_fn, normalizer_fn=norm_fn,
                    padding='SAME', weights_initializer=weights_init)
    return net
    