# Image segmentation models in TensorFlow.

"""
MIT License

Copyright (c) 2018 Sean M. Hendryx

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.image import ResizeMethod

#from losses import soft_dice_loss

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)


class UnetModel:
    """
    U-net model for image segmentation. Changes from the original u-net paper include batch normalization for faster learning and 
    bilinear upsampling in the decoding layers for ouput in the same shape as the input.
    Original Paper: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.num_classes = num_classes
        self.n_pix_rows = 572
        self.n_pix_cols = 572
        self.num_channels = 3

        self.input_ph = tf.placeholder(tf.float32, shape=(None, self.n_pix_rows, self.n_pix_cols, self.num_channels), name='X') # tf format tensor: (num_examples, dim_x, dim_y, num_channels)
        self.label_ph = tf.placeholder(tf.float32, shape=(None,self.n_pix_rows, self.n_pix_cols, self.num_classes), name='Y')

        self._he_init = tf.contrib.layers.variance_scaling_initializer()

       # Encode:
        conv1 = self.stack_encoder(self.input_ph, 64)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        
        conv2 = self.stack_encoder(pool1, 128)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        conv3 = self.stack_encoder(pool2, 256)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)    

        conv4 = self.stack_encoder(pool3, 512)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)    

        conv5 = tf.layers.conv2d(pool4, 1024, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self._he_init, name='Conv5')
        conv5 = tf.layers.batch_normalization(conv5, training=True)
        conv5 = tf.layers.conv2d(conv5, 1024, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self._he_init, name='Conv5-2')
        conv5 = tf.layers.batch_normalization(conv5, training=True)
        
        # Decode:
        conv6 = self.stack_decoder(conv5, skip_connection_tensor = conv4)
        conv7 = self.stack_decoder(conv6, conv3)
        conv8 = self.stack_decoder(conv7, conv2)
        conv9 = self.stack_decoder(conv8, conv1)
        
        # Final 1x1 convolution:
        self.logits = tf.layers.conv2d(conv9, self.num_classes, [1, 1], padding="SAME", activation=None, kernel_initializer=self._he_init, name='Output')
        
        # Get flat logits:
        self.flat_logits = tf.reshape(self.logits, (-1, int(np.prod(self.logits.get_shape()[1:]))))
        # Pixel-wise softmax:
        with tf.name_scope('predictions'):
            self.predictions = tf.nn.softmax(self.logits)
        self.iou = self._iou(self.label_ph, self.predictions)

        # Define loss:
        self.flat_label_ph = tf.reshape(self.label_ph, (-1, int(np.prod(self.label_ph.get_shape()[1:]))))
        # Cross entropy loss:
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.flat_label_ph, logits = self.flat_logits))
        # Soft dice loss as an approximation to IoU:
        #self.loss = soft_dice_loss(output = self.logits, target = self.label_ph)
        
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

    def stack_encoder(self, input_tensor, output_num_feature_channels:int): # -> tensor
        """
        Performs one encoding stack of U-net but with batch norm
        """
        conv = tf.layers.conv2d(input_tensor, output_num_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self._he_init)#, name='conv-{}-0'.format(output_num_feature_channels))
        conv = tf.layers.batch_normalization(conv, training=True) # could optionally apply bn before nonlinearity following original bn paper. To do this, use linear activation in conv2d, then batch norm, then tf.nn.relu
        conv = tf.layers.conv2d(conv, output_num_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self._he_init)#, name='conv{}-1'.format(output_num_feature_channels))
        return tf.layers.batch_normalization(conv, training=True)

    def stack_decoder(self, input_tensor, skip_connection_tensor): # -> tensor
        """
        Performs one decoding stack of U-net but with batch norm and bilinear upsampling to keep image sizes the same.
        """
        # Set number of feature maps equal to the number of channels/maps in skip_connection_tensor:
        output_num_feature_channels = skip_connection_tensor.get_shape()[3].value
        up = tf.layers.conv2d_transpose(input_tensor, output_num_feature_channels, [3, 3], strides=2, padding="SAME")#, name='Up')
        up = tf.layers.batch_normalization(up, training=True)

        # Bilinear upscale so that up can be concatenated with skip_connection_tensor so that it can be concatenated:
        new_height = skip_connection_tensor.get_shape()[1].value
        new_width = skip_connection_tensor.get_shape()[2].value
        upsample_size = tf.Variable([new_height, new_width], tf.int32)
        up = tf.image.resize_images(
            up,
            upsample_size, # size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size for the images.
            method=ResizeMethod.BILINEAR,
            align_corners=True
        )

        up = tf.concat([up, skip_connection_tensor], 3)
        conv = tf.layers.conv2d(up,  output_num_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self._he_init)#, name='Conv')
        conv = tf.layers.batch_normalization(conv, training=True)
        conv = tf.layers.conv2d(conv,  output_num_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self._he_init)#, name='Conv-2')
        return  tf.layers.batch_normalization(conv, training=True)

    def _iou(self, Y_true, Y_hat, epsilon=1e-7):
        """
        Returns an approximate intersection over union score

        intesection = Y_hat.flatten() * Y_true.flatten()
        IOU = 2 * intersection / (Y_hat.sum() + Y_true.sum() + epsilon) + epsilon

        :param y_hat: (4-D array): (N, H, W, 1)
        :param Y_true: (4-D array): (N, H, W, 1)
        :return: floating point IoU score
        """
        height, width, _ = Y_hat.get_shape().as_list()[1:]

        pred_flat = tf.reshape(Y_hat, [-1, height * width])
        true_flat = tf.reshape(Y_true, [-1, height * width])

        intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + epsilon
        denominator = tf.reduce_sum(
            pred_flat, axis=1) + tf.reduce_sum(
                true_flat, axis=1) + epsilon

        return tf.reduce_mean(intersection / denominator)

