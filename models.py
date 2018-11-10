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


class UnetModel:
    """
    U-net model for image segmentation. Changes from the original u-net paper include batch normalization for faster learning and 
    bilinear upsampling in the decoding layers for ouput in the same shape as the input.
    """
    def __init__(self, num_classes=1, n_pix_rows=512, n_pix_cols=512, n_unet_encoding_stacks=4, optimizer=tf.train.AdamOptimizer, **optim_kwargs):
        """
        :param num_classes: number of segmentation classes (i.e. number of channels in the masks)
        :param n_pix_rows: number of rows in the images and masks
        :param n_pix_cols: number of columns in the images and masks
        :param n_unet_encoding_stacks: number of encoding and decoding stacks in the u-net architecture
        """
        self.num_classes = num_classes
        self.num_output_channels = 2 if self.num_classes == 1 else self.num_classes
        self.n_pix_rows = n_pix_rows
        self.n_pix_cols = n_pix_cols
        self.n_unet_encoding_stacks = n_unet_encoding_stacks
        self.num_channels = 3
        self.batch_normalization_training_bool = True # argument to training parameter in tf.layers.batch_normalization
        self.weights_initializer = tf.glorot_normal_initializer()
        self.loss_function_name =  'soft_iou' # or 'cross_entropy'

        if n_unet_encoding_stacks < 1:
            raise ValueError('n_unet_encoding_stacks must be at least 1 but is: {}'.format(n_unet_encoding_stacks))

        self.input_ph = tf.placeholder(tf.float32, shape=(None, self.n_pix_rows, self.n_pix_cols, self.num_channels), name='X') # tf format tensor: (num_examples, dim_x, dim_y, num_channels)
        self.label_ph = tf.placeholder(tf.float32, shape=(None,self.n_pix_rows, self.n_pix_cols, self.num_output_channels), name='Y')

        # Encode:
        self.encoded, self.skip_connection_tensors = self.encode()

        # Decode:
        self.decoded = self.decode()

        # Final 1x1 convolution:
        self.logits = tf.layers.conv2d(self.decoded, self.num_output_channels, [1, 1], padding="SAME", activation=None, name='logits', kernel_initializer=self.weights_initializer)
        print('logits: {}'.format(self.logits))
        
        # Get flat logits:
        self.flat_logits = tf.reshape(self.logits, [-1, self.num_output_channels])
        print('flat_logits: {}'.format(self.flat_logits))
        # Pixel-wise softmax:
        with tf.name_scope('predictions'):
            self.predictions = tf.nn.softmax(self.logits)
        self.iou = self._iou(self.label_ph, self.predictions)

        # Define losses:
        print('labels: {}'.format(self.label_ph))
        self.flat_label_ph = tf.reshape(self.label_ph, [-1, self.num_output_channels])
        print('flat_labels: {}'.format(self.flat_label_ph))
        
        # Cross entropy loss:
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.flat_label_ph, logits = self.flat_logits))
        # Soft IoU/Jaccard loss as an approximation to IoU:
        self.soft_iou_loss = self._soft_iou_loss(self.label_ph, self.predictions)
        
        if self.loss_function_name == 'cross_entropy':
            self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)
        elif self.loss_function_name =='soft_iou':
            self.minimize_op = optimizer(**optim_kwargs).minimize(self.soft_iou_loss)
        else:
            raise ValueError('Loss string must be cross_entropy or soft_iou.')



    def encode(self):
        encodings = []
        for i in range(self.n_unet_encoding_stacks):
            if i == 0:
                conv = self.stack_encoder(self.input_ph, 2 ** 3)
            else:
                conv = self.stack_encoder(conv, 2 ** (3 + i))
            encodings.append(conv)
            # Max pool to compress image size (in rows & cols) in half:
            conv = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

        # Bottom of the U:
        n_feature_channels = 2 ** (3 + self.n_unet_encoding_stacks)
        conv = tf.layers.conv2d(conv, n_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self.weights_initializer)
        conv = tf.layers.conv2d(conv, n_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self.weights_initializer)
        conv = self.batch_normalization(conv)
        
        print('Skip connection tensors:')
        print(encodings)
        return conv, encodings

    def stack_encoder(self, input_tensor, output_num_feature_channels:int): # -> tensor
        """
        Performs one encoding stack of U-net but with batch norm
        """
        conv = tf.layers.conv2d(input_tensor, output_num_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self.weights_initializer)
        conv = tf.layers.conv2d(conv, output_num_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self.weights_initializer)
        conv = self.batch_normalization(conv)
        return conv

    def decode(self):
        """
        U-net upsampling
        
        """
        for i, skip_t in enumerate(reversed(self.skip_connection_tensors)):
            if i == 0:
                up_conv = self.stack_decoder(self.encoded, skip_t)
            else:
                up_conv = self.stack_decoder(up_conv, skip_t)
        return up_conv

    def stack_decoder(self, input_tensor, skip_connection_tensor): # -> tensor
        """
        Performs one decoding stack of U-net but with batch norm and bilinear upsampling to keep image sizes the same.
        """
        # Set number of feature maps equal to the number of channels/maps in skip_connection_tensor:
        output_num_feature_channels = skip_connection_tensor.get_shape()[3].value
        up = tf.layers.conv2d_transpose(input_tensor, output_num_feature_channels, [3, 3], strides=2, padding="SAME", activation=tf.nn.relu, kernel_initializer=self.weights_initializer)#, name='Up')
        up = self.batch_normalization(up)

        old_height = up.get_shape()[1].value
        old_width = up.get_shape()[2].value
        new_height = skip_connection_tensor.get_shape()[1].value
        new_width = skip_connection_tensor.get_shape()[2].value
        # Bilinear upsample `input_tensor` if up and skip_connection_tensor image dims are not equal (If in put image dims are power of 2, this condition should not be entered):
        if (old_height != new_height) or (old_width != new_width):
            # Bilinear upscale so that up can be concatenated with skip_connection_tensor:
            upsample_size = tf.Variable([new_height, new_width], tf.int32)
            up = tf.image.resize_images(
                up,
                upsample_size, # size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size for the images.
                method=ResizeMethod.BILINEAR,
                align_corners=True)

        up = tf.concat([skip_connection_tensor, up], 3)
        conv = tf.layers.conv2d(up,  output_num_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self.weights_initializer)
        conv = tf.layers.conv2d(conv,  output_num_feature_channels, [3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=self.weights_initializer)
        return self.batch_normalization(conv)

    def batch_normalization(self, tensor, batch_normalize=True):
        """
        Batch normalization wrapper
        """
        if batch_normalize:
            return tf.layers.batch_normalization(tensor, momentum = 0.9, training=self.batch_normalization_training_bool)
            #return tf.layers.batch_normalization(tensor, training=self.batch_normalization_training_bool)
        else:
            return tensor

    def _soft_iou_loss(self, Y_true, Y_hat):
        return -1*tf.math.log(self._iou(Y_true, Y_hat))

    def _iou(self, Y_true, Y_hat, epsilon=1e-7):
        """
        Returns an approximate intersection over union score for binary segmentation problems

        intesection = Y_hat.flatten() * Y_true.flatten()
        IOU = intersection / (Y_hat.sum() + Y_true.sum() + epsilon) + epsilon

        :param y_hat: (4-D array): (N, H, W,2)
        :param Y_true: (4-D array): (N, H, W,2)
        :return: floating point IoU score
        """
        Y_true_single_channel = Y_true[:,:,:,1]
        Y_hat_single_channel = Y_hat[:,:,:,1]
        print("_iou tensors:")
        height, width, _ = Y_hat.get_shape().as_list()[1:]

        pred_flat = tf.reshape(Y_hat_single_channel, [-1, height * width])
        true_flat = tf.reshape(Y_true_single_channel, [-1, height * width])

        print('pred_flat: {}'.format(pred_flat))
        print('true_flat: {}'.format(true_flat))

        intersection = tf.reduce_sum(pred_flat * true_flat, axis=1) + epsilon
        denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) - intersection + epsilon

        return tf.reduce_mean(intersection / denominator)