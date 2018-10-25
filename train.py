#Trains U-Net for image segmentation when provided with directory of images and corresponding masks.

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

# Usage: python train.py -i /data/images/ -m /data/masks/

import time
import os
import sys
import numpy as np
import tensorflow as tf
import random
from PIL import Image
import argparse

from utils.SegmentationDataLoader import SegmentationDataLoader
from models import UnetModel

# DEV:
import pdb
VIZTRAININGDATA = False # If True only visualize data from the data loader but do not perform train ops
DEBUG = False # If True, fetch intermediate tensors for begubbing and maybe visualize predictions
VIZPREDICTIONS = False # Only works if DEBUG also set to True
VERBOSE = False


def parse_args():
    """
    Returns an argument parser object for image segmentation training script.
    """
    parser = argparse.ArgumentParser(description='Trains U-Net for image segmentation when provided with directories of images and corresponding masks.')
    parser.add_argument('--images_dir', '-i', help='Path to directory containing images.', required=True)
    parser.add_argument('--masks_dir', '-m', help='Path to directory containing masks. Each element of an image-mask pair should have the same basename (regardless of extension).', required=True)
    parser.add_argument('--num_classes', help='Number of classes in segmentation task. 1 for binary segmentation.', default=1, type=int)
    parser.add_argument('--checkpoint', help='Checkpoint directory to write to (or restore from).', default='model_checkpoint', type=str)
    parser.add_argument('--restore_from_checkpoint', help='Boolean: whether or not to restore from checkpoint.', default=False, type=bool)
    parser.add_argument('--log_dir', help='Logging directory for tensorboard', default='logs/')
    parser.add_argument('--seed', help='Random seed', default=0, type=int)
    parser.add_argument('--batch_size', help='Training batch size', default=16, type=int)
    parser.add_argument('--epochs', help='Number of training epochs', default=50, type=int)
    parser.add_argument('--eval_interval', help='Training steps per evaluation', default=10, type=int)
    return parser.parse_args()

def main():
    # Args: 
    args = parse_args()
    images_dir=args.images_dir
    masks_dir=args.masks_dir
    num_classes=args.num_classes
    checkpoint_dir=args.checkpoint
    restore_from_ckpnt_bool=args.restore_from_checkpoint
    log_dir=args.log_dir

    # Training params:
    random.seed(args.seed)
    batch_size=args.batch_size
    epochs=args.epochs
    eval_interval=args.eval_interval


    mkdir(log_dir)

    # good hyperparameters:
    #optim_kwargs = {'learning_rate':0.0001} # with batch size of 16 and tf.train.RMSPropOptimizer
    optim_kwargs = {'learning_rate':0.0005} # with batch size of 16 and tf.train.AdamOptimizer
    model = UnetModel(num_classes=num_classes, n_unet_encoding_stacks=4, **optim_kwargs)
    X = model.input_ph
    Y = model.label_ph
    Y_hat = model.predictions
    cross_entropy_loss = model.loss
    soft_dice_loss = model.soft_dice_loss
    train_op = model.minimize_op
    iou_op = model.iou
    if DEBUG:
        logits = model.logits
        encoded = model.encoded
        decoded = model.decoded

    # Make a queue of filenames including all the jpg image and png mask files:
    image_paths, mask_paths = get_image_label_paths(images_dir, masks_dir)
    num_train = len(image_paths)
    if num_train != len(mask_paths):
        raise ValueError('num images != num masks.')

    # Split into training and test
    image_paths, mask_paths, test_image_paths, test_mask_paths = split_dataset(image_paths, mask_paths)

    # Initialize the segmentation data loading iterator for streaming batches of images and masks
    with tf.device('/cpu:0'):
        seg_data_loader = SegmentationDataLoader(image_paths=image_paths,
                                        mask_paths=mask_paths,
                                        image_extension='.jpg',
                                        image_channels=3,
                                        mask_channels=1,
                                        palette=[0, 255])
        batch_stream, init_op = seg_data_loader.data_batch(shuffle=False,
                                                            one_hot_encode=True,
                                                            batch_size=batch_size,
                                                            num_threads=1,
                                                            buffer=16)

        test_seg_data_loader = SegmentationDataLoader(image_paths=test_image_paths,
                                        mask_paths=test_mask_paths,
                                        image_extension='.jpg',
                                        image_channels=3,
                                        mask_channels=1,
                                        palette=[0, 255])
        test_batch_stream, test_init_op = test_seg_data_loader.data_batch(shuffle=False,
                                                            one_hot_encode=True,
                                                            batch_size=batch_size,
                                                            num_threads=1,
                                                            buffer=16)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Initialize the data queue
        sess.run(init_op)
        sess.run(test_init_op)

        saver, checkpoint_dir=setup_checkpointing(sess, checkpoint_dir, restore_from_ckpnt_bool)

        merged, train_writer, test_writer = setup_logging_and_get_merged_summaries(sess, cross_entropy_loss, iou_op)

        for epoch in range(epochs):
            print('Epoch: ', epoch)
            for step in range(0, num_train, batch_size):
                print('Samples seen this epoch: ', step)

                # Get training and test data:
                try:
                    train_image_batch, train_mask_batch = sess.run(batch_stream)
                except tf.errors.OutOfRangeError:
                    sess.run(init_op)
                    train_image_batch, train_mask_batch = sess.run(batch_stream)
                try:
                    test_image_batch, test_mask_batch = sess.run(test_batch_stream)
                except tf.errors.OutOfRangeError:
                    sess.run(test_init_op)
                    test_image_batch, test_mask_batch = sess.run(test_batch_stream)

                if VERBOSE:
                    print('train_image_batch.shape: ', train_image_batch.shape)

                if VIZTRAININGDATA: # DO NOT TRAIN
                    for i in range(train_image_batch.shape[0]):
                        print('unique values in train masks:')
                        print(np.unique(train_mask_batch[i,:,:,1]))
                        plot_two_images(train_image_batch[i,:], train_mask_batch[i,:,:,1])
                        plot_two_images(train_mask_batch[i,:,:,0], train_mask_batch[i,:,:,1])
                        
                else:
                    if ((step/batch_size) % eval_interval == 0):  # Record summaries and test-set IoU
                        summary, iou, test_loss, test_predictions = sess.run([merged, iou_op, cross_entropy_loss, Y_hat], 
                            feed_dict={
                                X: test_image_batch, 
                                Y: test_mask_batch})
                        print('test_loss: {}'.format(test_loss))
                        test_writer.add_summary(summary, step)
                        print('IoU on test-set at step {}: {}'.format(step, iou))
                        saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))
                    else:  # Record train set summaries, and take training step with train)_op
                        if DEBUG:
                            summary, _, train_loss, train_predictions, encoded_out, decoded_out, logits_out = sess.run([merged, train_op, cross_entropy_loss, Y_hat, encoded, decoded, logits], 
                                feed_dict={
                                    X: train_image_batch,
                                    Y: train_mask_batch})
                            print('train_loss: {}'.format(train_loss))
                            print('train_predictions: {}'.format(train_predictions))
                            if VIZPREDICTIONS:
                                for i in range(train_predictions.shape[0]):
                                    plot_two_images(train_image_batch[i,:], train_mask_batch[i,:,:,1])
                                    plot_two_images(train_mask_batch[i,:,:,1], train_predictions[i,:,:,1])
                        else:
                            summary, _, iou,sdl, train_loss = sess.run([merged, train_op, iou_op,soft_dice_loss, cross_entropy_loss], 
                                feed_dict={
                                    X: train_image_batch,
                                    Y: train_mask_batch})
                            print('iou: {}'.format(iou))
                            print('soft_dice_loss: {}'.format(sdl))
                            print('train_loss: {}'.format(train_loss))
                            train_writer.add_summary(summary, step)

        saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))



def setup_checkpointing(sess, checkpoint_dir, restore_from_ckpnt_bool=True):
    """
    Sets up tf training saver, checkpointing dir, and optionally restores checkpoint to continue training.
    """
    saver = tf.train.Saver()
    if restore_from_ckpnt_bool and os.path.exists(checkpoint_dir) and tf.train.checkpoint_exists(
            checkpoint_dir):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, latest_checkpoint)
    else:
        try:
            os.rmdir(checkpoint_dir)
        except FileNotFoundError:
            pass
        os.mkdir(checkpoint_dir)
    return saver, checkpoint_dir

def setup_logging_and_get_merged_summaries(sess, cross_entropy_tensor, iou_tensor, log_dir='logs'):
    mkdir(log_dir)
    #loss_tensor:
    tf.summary.scalar('cross_entropy', cross_entropy_tensor)

    with tf.name_scope('IoU'):
        tf.summary.scalar('IoU', iou_tensor)

    # Merge all the summaries and write them out to
    # 'logs' (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    tf.global_variables_initializer().run()

    return merged, train_writer, test_writer


def split_dataset(image_paths, mask_paths, train_percentage=0.8):
    """
    Splits the dataset into a training and test set.
    image and mask paths are assumed to be in corresponding order.
    :return: train_image_paths, train_mask_paths, test_image_paths, test_mask_paths
    """
    if train_percentage > 0.0:
        all_data = list(zip(image_paths, mask_paths))
        random.shuffle(all_data)
        num_train = int(np.floor(len(image_paths) * train_percentage))
        train_image_paths = [x[0] for x in all_data[:num_train]]
        train_mask_paths = [x[1] for x in all_data[:num_train]]
        test_image_paths = [x[0] for x in all_data[num_train:]]
        test_mask_paths = [x[1] for x in all_data[num_train:]]
        #zip(*all_data[:num_train]), zip(*all_data[num_train:])
        print("Dataset split into {} training examples and {} test examples".format(len(train_mask_paths), len(test_mask_paths)))
        return train_image_paths, train_mask_paths, test_image_paths, test_mask_paths
    else:
        return image_paths, mask_paths, None, None

def get_test_data(image_paths, mask_paths, test_percentage=0.2):
    import random
    m = floor(len(image_paths) * test_percentage)
    t1 = [L.pop(random.randrange(len(L))) for _ in range(m)]
    x_sub, y_sub = zip(*random.sample(list(zip(image_paths, mask_paths)), m))

def get_image_label_paths(images_dir, labels_dir, image_ext = '.jpg', label_ext = '.png', verbose = True):
    """
    Returns the full (not neccesarily absolute) paths to the images and corresponding labels
    by matching basenames without extensions.
    :param images_dir: path to directory containing images
    :param labels_dir: path to directory containing pixel-level image mask labels
    :return: list of image paths and label paths. Only returns those images for which there is a corresponding image. Assumes every label has a corresponding image
    """
    print('Getting corresponding image and label paths from dirs: {} {}.'.format(images_dir, labels_dir))
    image_basenames_noext = [remove_ext(os.path.basename(x)) for x in os.listdir(images_dir) if x.endswith(image_ext)]
    label_basenames_noext = [remove_ext(os.path.basename(x)) for x in os.listdir(labels_dir) if x .endswith(label_ext)]

    # Get intersection:
    basenames_noext = set(image_basenames_noext).intersection(label_basenames_noext)

    # Get image_paths for which there is a corresponsing label: i.e., only train on image-label pairs (regardless of extension):
    image_paths = []
    label_paths = []
    for basename_noext in basenames_noext:
        image_paths.append(os.path.join(images_dir, basename_noext + image_ext))
        label_paths.append(os.path.join(labels_dir, basename_noext + label_ext))

    if verbose:
        print('Found {} labels:'.format(len(label_paths)))
        #print(label_paths)
        print('and {} corresponding images:'.format((len(image_paths))))
        #print(image_paths)
    return image_paths, label_paths

def listdir_fullpath(d, extension = '.jpg'):
    """
    Returns the absolute paths of all files with extension in d
    :param d: path to directory d
    :param extension: list files with this extension
    :return: list of strings of absolute paths of files in d with extension `extension`
    """
    return [os.path.join(d, f) for f in os.listdir(d) if f.endswith(extension)]

def remove_ext(path):
    return os.path.splitext(path)[0]

def mkdir(path):
    """
    Recursive create dir at `path` if `path` does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def plot_two_images(A, B):
    import matplotlib.pyplot as plt


    plt.figure()

    plt.subplot(121)
    plt.imshow(A)
    plt.subplot(122)
    plt.imshow(B)

    plt.show()





if __name__ == '__main__':
    main()