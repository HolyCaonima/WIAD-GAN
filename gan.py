
import math
import customDataGeter
import model
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly

from scipy.misc import imsave

class GAN(object):

    def __init__(self, version, clip_abs, hidden_size, batch_size, learning_rate, data_directory, log_directory):
        '''GAN Construction function

        Args:
            hidden_size: the hidden size for random Value
            batch_size: the img num per batch
            learning_rate: the learning rate

        Returns:
            A tensor that expresses the encoder network

        Notify: output size dependence
        '''
        if version == 0:
            self.img_size = [32, 32]
        if version == 1:
            self.img_size = [64, 64]
        self.version = version
        self.clip_abs = clip_abs
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_directory = data_directory
        self.log_directory = log_directory

        # build the graph
        self.opt_g, self.opt_c, self.z, self.real_data = self._build_graph()
        self.merged_all = tf.summary.merge_all()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.log_directory, self.sess.graph)


    def _build_graph(self):
        z = tf.placeholder(tf.float32, shape=(self.batch_size, self.hidden_size))

        with tf.variable_scope('generator'):
            if self.version == 0:
                self.g_out = model.generator_conv(z, 3)
            if self.version == 1:
                self.g_out = model.generator_conv_v2(z, 3)

        #real_data = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size[0], self.img_size[1], 3))
        real_data = customDataGeter.input(self.data_directory, self.img_size, self.batch_size)

        if self.version == 0:
            true_logit = model.critic_conv(real_data, self.batch_size)
            fake_logit = model.critic_conv(self.g_out, self.batch_size, reuse=True)
        if self.version == 1:
            true_logit = model.critic_conv_v2(real_data, self.batch_size)
            fake_logit = model.critic_conv_v2(self.g_out, self.batch_size, reuse=True)

        # define the loss
        self.c_loss = tf.reduce_mean(fake_logit - true_logit)
        self.g_loss = tf.reduce_mean(-fake_logit)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)

        # img summary
        img_sum = tf.summary.image("img", self.g_out, max_outputs=10)

        # get the params
        theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        opt_g = ly.optimize_loss(loss=self.g_loss, learning_rate=self.learning_rate,
                    optimizer=tf.train.RMSPropOptimizer, 
                    variables=theta_g, global_step=counter_g)

        counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        opt_c = ly.optimize_loss(loss=self.c_loss, learning_rate=self.learning_rate,
                    optimizer=tf.train.RMSPropOptimizer, 
                    variables=theta_c, global_step=counter_c)

        # define the clip op
        clipped_var_c = [tf.assign(var, tf.clip_by_value(var, -self.clip_abs, self.clip_abs)) for var in theta_c]
        # merge the clip operations on critic variables
        with tf.control_dependencies([opt_c]):
            opt_c = tf.tuple(clipped_var_c)

        return opt_g, opt_c, z, real_data

    def _get_feed_noise(self):
        feed_noise = np.random.normal(0, 1, [self.batch_size, self.hidden_size]).astype(np.float32)
        return feed_noise
    
    def get_loss(self):
        feed_noise = self._get_feed_noise()
        return self.sess.run([self.c_loss, self.g_loss], feed_dict={self.z:feed_noise})

    def update_params(self, current_step, d_step = 5, g_step = 1):
        if current_step < 25 or current_step % 500 == 0:
            citers = 100
        else:
            citers = d_step
        # train citers 
        for j in range(citers):
            feed_noise = self._get_feed_noise()
            self.sess.run(self.opt_c, feed_dict={self.z:feed_noise})

        feed_noise = self._get_feed_noise()
        self.sess.run(self.opt_g, feed_dict={self.z:feed_noise})

    def generate_and_save_images(self, num_samples, directory):
        '''Generates the images using the model and saves them in the directory

        Args:
            num_samples: number of samples to generate
            directory: a directory to save the images

        Notify: output size dependence
        '''
        feed_noise = self._get_feed_noise()
        imsize = self.img_size
        im_w = int(math.ceil(math.sqrt(num_samples)))
        big_img = np.zeros([im_w*imsize[1],im_w*imsize[0],3])
        imgs = self.sess.run(self.g_out, feed_dict={self.z:feed_noise})
        for k in range(imgs.shape[0]):
            big_img[(k/im_w)*imsize[1]:((k/im_w)+1)*imsize[1], (k%im_w)*imsize[0]:((k%im_w)+1)*imsize[0],:] = imgs[k].reshape(imsize[1], imsize[0], 3)
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder) 
            imsave(os.path.join(imgs_folder, '%d.png') % k, imgs[k].reshape(self.img_size[0], self.img_size[1], 3))
        imsave(os.path.join(imgs_folder,"Agg.png"), big_img)
    
    def get_merged_image(self, num_samples):
        feed_noise = self._get_feed_noise()
        imsize = self.img_size
        im_w = int(math.ceil(math.sqrt(num_samples)))
        big_img = np.zeros([im_w*imsize[1],im_w*imsize[0],3])
        imgs = self.sess.run(self.g_out, feed_dict={self.z:feed_noise})
        for k in range(imgs.shape[0]):
            big_img[(k/im_w)*imsize[1]:((k/im_w)+1)*imsize[1], (k%im_w)*imsize[0]:((k%im_w)+1)*imsize[0],:] = imgs[k].reshape(imsize[1], imsize[0], 3)
        big_img = big_img.reshape([1,big_img.shape[0],big_img.shape[1],3])
        return big_img