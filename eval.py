""" eval.py
    This script to evaluate trained model for semantic segmentation
    using mean intersection over union.
"""

import os
import sys
import time
import math
import logging
import importlib
import numpy as np
import tensorflow as tf
from PIL import Image

from camvid import load_splited_path
from inputs import read_image_label_from_queue, scale_fixed_size
from utils import make_dirs, save_img, vis_semseg

flags = tf.app.flags
FLAGS = flags.FLAGS

# Basic arguments
flags.DEFINE_string('arch', 'segnet', 'Network architecure')
flags.DEFINE_string('outdir', 'output/camvid', 'Output directory')
flags.DEFINE_string('resdir', 'results', 'Directory to visualize prediction')

# Dataset arguments
flags.DEFINE_string('dataset', 'camvid', 'Dataset name')
flags.DEFINE_string('checkpoint_dir', 'output/trained_model/checkpoint_dir',
    'Directory where to read model checkpoint.')
flags.DEFINE_string('indir', 'data/CamVid', 'Dataset directory')

# Evaluation arguments
flags.DEFINE_integer('channel', 3, 'Channel of an input image')
flags.DEFINE_integer('num_class', 12, 'Number of class to classify')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('height', 224, 'Input height')
flags.DEFINE_integer('width', 224, 'Input width')
flags.DEFINE_integer('num_sample', 233, 'Number of test samples')


def build_model(images, phase_train):
    model = importlib.import_module(FLAGS.arch)
    logits = model.inference(images, phase_train)
    prob, pred = model.predict(logits)
    return prob


def crop_and_upsample(prob, resized_image, raw_image, mask):
    resized_h = tf.shape(resized_image)[1]
    resized_w = tf.shape(resized_image)[2]
    resized_shape = tf.stack([1, resized_h, resized_w, FLAGS.num_class])
    raw_shape = tf.shape(raw_image)[:2]
    cropped_prob = tf.boolean_mask(
        tf.squeeze(prob), tf.squeeze(tf.equal(mask, 0)))
    reshaped_prob = tf.reshape(cropped_prob, resized_shape)
    upsampled_prob = tf.image.resize_bilinear(reshaped_prob, raw_shape)
    return tf.squeeze(tf.cast(tf.argmax(upsampled_prob, axis=-1), tf.int32))


def evaluate(res_dir, ignore_label=255):
    logging.info('Evaluating: {}'.format(FLAGS.arch))
    logging.info('FLAGS: {}'.format(FLAGS.__flags))

    graph = tf.Graph()
    with graph.as_default():
        dataset = importlib.import_module(FLAGS.dataset)
        label_info = dataset.label_info
        cmap = label_info['cmap']

        txt_path = os.path.join(FLAGS.indir, 'test.txt')
        image_list, label_list = load_splited_path(txt_path, FLAGS.indir)

        raw_image, raw_label = read_image_label_from_queue(image_list, label_list)
        cropped_image, resized_image, mask = scale_fixed_size(
            raw_image, raw_label, [FLAGS.height, FLAGS.width])

        phase_train = tf.placeholder(tf.bool, name='phase_train')
        output = build_model(cropped_image, phase_train)
        upsampled_pred = crop_and_upsample(output, resized_image, raw_image, mask)

        pred = tf.reshape(upsampled_pred, [-1,])
        label = tf.reshape(raw_label, [-1,])
        weights = tf.cast(tf.not_equal(label, ignore_label), tf.int32)

        mean_iou, update_op = tf.contrib.metrics.streaming_mean_iou(
            pred, label, num_classes=FLAGS.num_class, weights=weights)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        logging.info('Start evaluating...')

        for i in range(FLAGS.num_sample):
            _img, _gt, _pred, _miou, _update_op = sess.run(
                [raw_image, raw_label, upsampled_pred, mean_iou, update_op],
                feed_dict={phase_train: False})

            img_name = '{}.png'.format(i + 1)
            save_img(_img, os.path.join(res_dir, 'img'), img_name)
            vis_semseg(_gt, cmap, os.path.join(res_dir, 'gt'), img_name)
            vis_semseg(_pred, cmap, os.path.join(res_dir, 'pred'), img_name)

            message = 'Evaluted {}'.format(res_dir + '/' + img_name)
            print(message)
            logging.info(message)

        result_message = 'mean_iou: {}'.format(mean_iou.eval())
        print(result_message)
        logging.info(result_message)

        logging.info('Finished.')
        coord.request_stop()
        coord.join(threads)


def main(_):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        meta_graph_path = ckpt.model_checkpoint_path + ".meta"
        step = os.path.splitext(os.path.basename(meta_graph_path))[0]
    else:
        print('No checkpoint file found')
        sys.exit()

    res_dir = os.path.join(FLAGS.outdir, FLAGS.resdir, step)
    make_dirs(res_dir)

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename='{}/eval.log'.format(res_dir),
        filemode='w', level=logging.INFO)

    evaluate(res_dir)


if __name__ == '__main__':
    tf.app.run()
