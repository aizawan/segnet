""" train.py
"""

import os
import sys
import time
import logging
import importlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from datetime import datetime

from utils import make_dirs
from inputs import read_and_decode

flags = tf.app.flags
FLAGS = flags.FLAGS

# Basic arguments
flags.DEFINE_string('arch', 'segnet', 'Network architecure')
flags.DEFINE_string('outdir', 'output/camvid', 'Output directory')

# Dataset arguments
flags.DEFINE_string('dataset', 'camvid', 'Dataset name')
flags.DEFINE_string('tfrecord',
    '/tmp/data/camvid/camvid-train.tfrecord', 'TFRecord path')

# Model arguments
flags.DEFINE_integer('channel', 3, 'Channel of an input image')
flags.DEFINE_integer('num_class', 12, 'Number of class to classify')
flags.DEFINE_integer('height', 224, 'Input height')
flags.DEFINE_integer('width', 224, 'Input width')

# Training arguments
flags.DEFINE_integer('batch_size', 5, 'Batch size')
flags.DEFINE_integer('iteration', 10000, 'Number of training iterations')
flags.DEFINE_integer('num_threads', 8, 'Number of threads to read batches')
flags.DEFINE_integer('min_after_dequeue', 10, 'min_after_dequeue')
flags.DEFINE_integer('seed', 1234, 'Random seed')
flags.DEFINE_integer('snapshot', 2000, 'Snapshot')
flags.DEFINE_integer('print_step', 1, 'Number of step to print training log')
flags.DEFINE_string('optimizer', 'sgd', 'optimizer')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_boolean('cb', False, 'Class Balancing')


np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)


def save_model(sess, saver, step, outdir, message):
    print('Saving...')
    saver.save(sess, outdir + '/model', global_step=step)
    logging.info(message)
    print(message)


def train(model_dir, summary_dir):
    logging.info('Training {}'.format(FLAGS.arch))
    logging.info('FLAGS: {}'.format(FLAGS.__flags))
    print(FLAGS.__flags)

    graph = tf.Graph()
    with graph.as_default():
        dataset = importlib.import_module(FLAGS.dataset)

        fn_queue = tf.train.string_input_producer([FLAGS.tfrecord])
        images, labels = read_and_decode(
            fn_queue=fn_queue,
            target_height=FLAGS.height,
            target_width=FLAGS.width,
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_threads,
            min_after_dequeue=FLAGS.min_after_dequeue,
            shuffle=True)

        phase_train = tf.placeholder(tf.bool, name='phase_train')

        model = importlib.import_module(FLAGS.arch)
        logits = model.inference(images, phase_train)
        acc = model.acc(logits, labels)

        if FLAGS.cb:
            loss = model.loss(logits, labels, cb=dataset.label_info['cb'])
        else:
            loss = model.loss(logits, labels)

        summary = model.setup_summary(loss, acc)

        train_op = model.train_op(loss, FLAGS.optimizer,
            lr=FLAGS.learning_rate, momentum=FLAGS.momentum)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        start_time = time.time()

        step = 0
        logging.info('Start training...')
        try:
            while not coord.should_stop():
                feed_dict = {phase_train: True}
                _, loss_value, acc_value, summary_str = sess.run(
                    [train_op, loss, acc, summary], feed_dict=feed_dict)

                duration = time.time() - start_time
                message = \
                    'arch: {} '.format(FLAGS.arch) + \
                    'step: {} '.format(step + 1) + \
                    'loss: {:.3f} '.format(loss_value) + \
                    'acc: {:.3f} '.format(acc_value) + \
                    'duration: {:.3f}sec '.format(duration) + \
                    'time_per_step: {:.3f}sec'.format(duration / (step + 1))

                writer.add_summary(summary_str, step)

                if not step % FLAGS.print_step:
                    print(message)
                    logging.info(message)

                if not step % FLAGS.snapshot and not step == 0:
                    message = 'Done training for {} steps.'.format(step)
                    save_model(sess, saver, step, model_dir, message)

                if step == FLAGS.iteration: break

                step += 1

        except KeyboardInterrupt:
            coord.request_stop()

        finally:
            coord.request_stop()

        coord.join(threads)


def main(_):
    outdir = os.path.join(FLAGS.outdir, FLAGS.arch)
    trained_model = os.path.join(outdir, 'trained_model')
    summary_dir = os.path.join(outdir, 'summary')

    make_dirs(trained_model)

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename='{}/train.log'.format(outdir),
        filemode='w', level=logging.INFO)

    train(trained_model, summary_dir)


if __name__ == '__main__':
    tf.app.run()
