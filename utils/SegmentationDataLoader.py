# Class to stream batches of images and masks.

"""
MIT License

Copyright (c) 2017 Hasnain Raza
Modified Work Copyright (c) 2018 Sean M. Hendryx

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

# USAGE:
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from SegmentationDataLoader import SegmentationDataLoader
import os

plt.ioff()

IMAGE_DIR_PATH = 'data/training/images'
MASK_DIR_PATH = 'data/training/masks'

# create list of PATHS
image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.png')]

# Where image_paths[0] = 'data/training/images/image_0.png' 
# And mask_paths[0] = 'data/training/masks/mask_0.png'


# Initialize the segmentation data loading iterator object for streaming batches of images and masks
seg_data_loader = SegmentationDataLoader(image_paths=image_paths,
                                mask_paths=mask_paths,
                                image_extension='.jpg',
                                image_channels=3,
                                mask_channels=1,
                                palette=[0, 255])
batch_stream, init_op = seg_data_loader.data_batch(shuffle=False,
                                                    one_hot_encode=True,
                                                    batch_size=16,
                                                    num_threads=2,
                                                    buffer=32)

with tf.Session() as sess:
    # Initialize the data queue
    sess.run(init_op)
    # Evaluate the tensors
    train_image_batch, train_mask_batch = sess.run(batch_stream)
                                 
# Do whatever you want now, like creating a feed dict and train your models,
# You can also directly feed in the tf tensors to your models to avoid using a feed dict.
"""


import tensorflow as tf
import random


class SegmentationDataLoader(object):

    def __init__(self, image_paths, mask_paths, image_extension='.jpg', mask_extension='.png',
                    image_channels=3, mask_channels=1, palette=[0,255],  seed=None):
        """
        Initializes the segmentation data loader object.
        :param image_paths: List of paths of training images.
        :param mask_paths: List of paths of training masks (segmentation masks)
        :param image_extension: The file format of images, either '.jpg' or '.png'.
        :param image_channels: Int number of channels in the images.
        :param mask_channels: Int number of channels in the masks.
        :param palette: A list of pixel values in the mask, the index of a value
            in palette becomes the channel index of the value in mask.
            For example, all if mask is binary (0, 1), then palette should
            be [0, 1], mask will then have depth 2, where the first index along depth
            will be 1 where the original mask was 0, and the second index along depth will
            be 1 where the original mask was 1. Works for and rgb palette as well,
            specify the palette as: [[255,255,255], [0, 255, 255]] etc.
            (one hot encoding).
        :param seed: An int, if not specified, chosen randomly. Used as the seed for the RNG in the 
            data pipeline.

        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        # Convert lists of paths to tensors for tensorflow
        self.image_paths_tensor = tf.constant(self.image_paths)
        self.mask_paths_tensor = tf.constant(self.mask_paths)
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.palette = palette
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed


    def _normalize_data(self, image, mask):
        """
        Normalize image and mask within range 0-1.
        """
        image = tf.cast(image, tf.float32)
        image = image / 255.0

        return image, mask


    def _parse_data(self, image_paths, mask_paths):
        """
        Reads image and mask files depending on
        specified exxtension.
        """
        image_content = tf.read_file(image_paths)
        mask_content = tf.read_file(mask_paths)

        if self.image_extension == '.jpg':
            images = tf.image.decode_jpeg(image_content, channels=self.image_channels)
        elif self.image_extension == '.png':
            images = tf.image.decode_png(image_content, channels=self.image_channels)
        else:
            raise ValueError("Specified image extension is not supported,\
                              please use either jpg or png images")

        if self.mask_extension == '.png':
            masks = tf.image.decode_png(mask_content, channels=self.mask_channels)
        elif self.mask_extension == '.jpg':
            masks = tf.image.decode_jpeg(mask_content, channels=self.mask_channels)
        else:
            raise ValueError("Specified image extension is not supported,\
                              please use either jpg or png images")

        return images, masks


    def _one_hot_encode(self, image, mask):
        """
        Converts mask to a one-hot encoding specified by the semantic map.
        """
        one_hot_map = []
        for colour in self.palette:
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)
        
        return image, one_hot_map

    def data_batch(self, shuffle=False, one_hot_encode=True, batch_size=16, num_threads=2, buffer=32):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        :param shuffle: Boolean, whether to shuffle data in buffer or not.
        :param batch_size: Number of images/masks in each batch returned.
        :param one_hot_encode: Boolean, whether to one hot encode the mask image or not.
            Encoding will done according to the palette specified when
            initializing the object.
        :param num_threads: Number of parallel subprocesses to load data.
        :param buffer: Number of images to prefetch in buffer.
        :return:
            next_element: A tensor with shape [2], where next_element[0]
                          is image batch, next_element[1] is the corresponding
                          mask batch.
            init_op: Data initializer op, needs to be executed in a session
                     for the data queue to be filled up and the next_element op
                     to yield batches.
        """

        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices(
            (self.image_paths_tensor, self.mask_paths_tensor))

        # Parse images and labels
        data = data.map(
            self._parse_data, num_parallel_calls=num_threads).prefetch(buffer)

        # One hot encode the mask
        if one_hot_encode:
            if self.palette is None:
                raise ValueError('No Palette for one-hot encoding specified in the data loader!')
            data = data.map(self._one_hot_encode, num_parallel_calls=num_threads).prefetch(buffer)

        # Batch the data
        data = data.batch(batch_size)

        # Normalize
        data = data.map(self._normalize_data,
                        num_parallel_calls=num_threads).prefetch(buffer)

        if shuffle:
            data = data.shuffle(buffer)

        # Create iterator
        iterator = tf.data.Iterator.from_structure(
            data.output_types, data.output_shapes)

        # Next element Op
        next_element = iterator.get_next()

        # Data set init. op
        init_op = iterator.make_initializer(data)

        return next_element, init_op