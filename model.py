import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
import ops

def generator_conv(z, channel):
    train = ly.fully_connected(z, 4 * 4 * 512, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, (-1, 4, 4, 512))
    train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 128, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 64, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, channel, 3, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    return train

def generator_mlp(z, channel):
    train = ly.fully_connected(
        z, 4 * 4 * 512, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    train = ly.fully_connected(
        train, 1024, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    train = ly.fully_connected(
        train, 1024, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    train = ly.fully_connected(
        train, 32*32*channel, activation_fn=tf.nn.tanh, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, tf.stack([z.get_shape()[0], 32, 32, channel]))
    return train

def generator_conv_v2(input_tensor, channel):
    
    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)

    #net = ly.fully_connected(net, 4*4*1024, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    #net = ly.fully_connected(net, 1024, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    #net = ly.fully_connected(net, 1024, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    net = ly.fully_connected(net, 4*4*1024, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    # reshape for conv
    net = tf.reshape(net, [-1, 4, 4, 1024])

    net = ly.conv2d_transpose(net, 512, 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    net = ly.conv2d_transpose(net, 256, 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    net = ly.conv2d_transpose(net, 128, 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    net = ly.conv2d_transpose(net, 64, 5, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    net = ly.conv2d_transpose(net, channel, 5, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    '''
    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)

    net = ly.fully_connected(net, 4*4*1024, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
    net = tf.reshape(net, [-1, 4, 4, 1024])
    net = ly.conv2d(net, 1024, 2, stride=1,
                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    net = ops.resize_conv2d(net, 512, 5, [8, 8])
    net = ops.resize_conv2d(net, 256, 5, [16, 16])
    net = ops.resize_conv2d(net, 128, 5, [32, 32])
    net = ops.resize_conv2d(net, 64, 5, [64, 64])
    net = ly.conv2d(net, channel, 5, stride=1,
                    activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    '''
    return net


def critic_conv(img, batch_size, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        size = 64
        img = ly.conv2d(img, num_outputs=size, kernel_size=3,
                        stride=2, activation_fn=ops.lrelu)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3,
                        stride=2, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3,
                        stride=2, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=3,
                        stride=2, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
        logit = ly.fully_connected(tf.reshape(
            img, [batch_size, -1]), 1, activation_fn=None)
    return logit

def critic_mlp(img, batch_size, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.fully_connected(tf.reshape(img, [batch_size, -1]), 1024, activation_fn=tf.nn.relu)
        img = ly.fully_connected(img, 1024, activation_fn=tf.nn.relu)
        img = ly.fully_connected(img, 1024, activation_fn=tf.nn.relu)
        logit = ly.fully_connected(img, 1, activation_fn=None)
    return logit

def critic_conv_v2(input_tensor, batch_size, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        # add some noise
        #net = input_tensor + tf.random_normal([batch_size, int(input_tensor.get_shape()[1]), int(input_tensor.get_shape()[2]), int(input_tensor.get_shape()[3])], 0, 0.1)
        size = 64
        net = ly.conv2d(input_tensor, num_outputs=size, kernel_size=5,
                        stride=2, activation_fn=ops.lrelu)
        net = ly.conv2d(net, num_outputs=size * 2, kernel_size=5,
                        stride=2, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
        net = ly.conv2d(net, num_outputs=size * 4, kernel_size=5,
                        stride=2, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
        net = ly.conv2d(net, num_outputs=size * 8, kernel_size=5,
                        stride=2, activation_fn=ops.lrelu, normalizer_fn=ly.batch_norm)
        # flaten the net for full connection layer
        net = ly.flatten(net)
        #net = ops.minibatch_disc(net)
        net = ly.fully_connected(net, 1, activation_fn=None)

    return net
