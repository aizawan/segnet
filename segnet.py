""" segnet.py
    Implementation of SegNet for Semantic Segmentation.
"""


import os
import sys
import time
import numpy as np
import tensorflow as tf

from ops import *

flags = tf.app.flags
FLAGS = flags.FLAGS


def inference(inputs, phase_train):
    with tf.variable_scope(FLAGS.arch):
        h, mask = encoder(inputs, phase_train, name='encoder')
        logits = decoder(h, mask, phase_train, name='decoder')
    return logits


def loss(logits, labels, ignore_label=-1, cb=None, name='loss'):
    with tf.name_scope(name):
        num_class = logits.get_shape().as_list()[-1]
        epsilon = tf.constant(value=1e-10)
        logits = tf.reshape(logits, (-1, num_class))
        labels = tf.reshape(labels, (-1, 1))
        not_ign_mask = tf.where(tf.not_equal(tf.squeeze(labels), ignore_label))

        logits = tf.reshape(tf.gather(logits, not_ign_mask), (-1, num_class))
        labels = tf.reshape(tf.gather(labels, not_ign_mask), (-1, 1))

        one_hot = tf.reshape(
            tf.one_hot(labels, depth=num_class), (-1, num_class))

        prob = tf.nn.softmax(logits)

        if cb is not None:
            xe = -tf.reduce_sum(
                tf.multiply(one_hot * tf.log(prob + epsilon), cb),
                reduction_indices=[1])
        else:
            xe = tf.nn.softmax_cross_entropy_with_logits(
                labels=one_hot, logits=logits)

        mxe = tf.reduce_mean(xe)
    return mxe


def acc(logits, labels, ignore_label=-1, name='acc'):
    with tf.name_scope(name):
        logits = tf.reshape(logits, (-1, FLAGS.num_class))
        labels = tf.reshape(labels, [-1])

        not_ign_mask = tf.where(tf.not_equal(tf.squeeze(labels), ignore_label))

        logits = tf.reshape(tf.gather(logits, not_ign_mask), (-1, FLAGS.num_class))
        labels = tf.reshape(tf.gather(labels, not_ign_mask), [-1])

        epsilon = tf.constant(value=1e-10, name="epsilon")
        logits = tf.add(logits, epsilon)

        prob = tf.nn.softmax(logits)
        pred = tf.cast(tf.argmax(prob, axis=1), tf.int32)

        correct_pred = tf.equal(pred, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy


def predict(logits, name='predict'):
    with tf.name_scope(name):
        prob = tf.squeeze(tf.nn.softmax(logits))
        pred = tf.squeeze(tf.cast(tf.argmax(prob, axis=-1), tf.int32))
    return prob, pred


def train_op(loss, opt_name, **kwargs):
    optimizer = _get_optimizer(opt_name, kwargs)
    return optimizer.minimize(loss)


def setup_summary(loss, acc):
    summary_loss = tf.summary.scalar('loss', loss)
    summary_acc = tf.summary.scalar('acc', acc)
    return tf.summary.merge([summary_loss, summary_acc])


def _get_optimizer(opt_name, params):
    if opt_name == 'adam':
        return tf.train.AdamOptimizer(params['lr'])
    elif opt_name == 'adadelta':
        return tf.train.AdadeltaOptimizer(params['lr'])
    elif opt_name == 'sgd':
        return tf.train.GradientDescentOptimizer(params['lr'])
    elif opt_name == 'momentum':
        return tf.train.MomentumOptimizer(params['lr'], params['momentum'])
    elif opt_name == 'rms':
        return tf.train.RMSPropOptimizer(params['lr'])
    elif opt_name == 'adagrad':
        return tf.train.AdagradOptimizer(params['lr'])
    else:
        print('error')


def n_enc_block(inputs, phase_train, n, k, name):
    h = inputs
    with tf.variable_scope(name):
        for i in range(n):
            h = conv2d(h, k, 3, stride=1, name='conv_{}'.format(i + 1))
            h = batch_norm(h, phase_train, name='bn_{}'.format(i + 1))
            h = relu(h, name='relu_{}'.format(i + 1))
        h, mask = maxpool2d_with_argmax(h, name='maxpool_{}'.format(i + 1))
    return h, mask


def encoder(inputs, phase_train, name='encoder'):
    with tf.variable_scope(name):
        h, mask_1 = n_enc_block(inputs, phase_train, n=2, k=64, name='block_1')
        h, mask_2 = n_enc_block(h, phase_train, n=2, k=128, name='block_2')
        h, mask_3 = n_enc_block(h, phase_train, n=3, k=256, name='block_3')
        h, mask_4 = n_enc_block(h, phase_train, n=3, k=512, name='block_4')
        h, mask_5 = n_enc_block(h, phase_train, n=3, k=512, name='block_5')
    return h, [mask_5, mask_4, mask_3, mask_2, mask_1]


def n_dec_block(inputs, mask, adj_k, phase_train, n, k, name):
    in_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name):
        h = maxunpool2d(inputs, mask, name='unpool')
        for i in range(n):
            if i == (n - 1) and adj_k:
                h = conv2d(h, k / 2, 3, stride=1, name='conv_{}'.format(i + 1))
            else:
                h = conv2d(h, k, 3, stride=1, name='conv_{}'.format(i + 1))
            h = batch_norm(h, phase_train, name='bn_{}'.format(i + 1))
            h = relu(h, name='relu_{}'.format(i + 1))
    return h


def dec_last_conv(inputs, phase_train, k, name):
    with tf.variable_scope(name):
        h = conv2d(inputs, k, 1, name='conv')
    return h


def decoder(inputs, mask, phase_train, name='decoder'):
    with tf.variable_scope(name):
        h = n_dec_block(inputs, mask[0], False, phase_train, n=3, k=512, name='block_5')
        h = n_dec_block(h, mask[1], True, phase_train, n=3, k=512, name='block_4')
        h = n_dec_block(h, mask[2], True, phase_train, n=3, k=256, name='block_3')
        h = n_dec_block(h, mask[3], True, phase_train, n=2, k=128, name='block_2')
        h = n_dec_block(h, mask[4], True, phase_train, n=2, k=64, name='block_1')
        h = dec_last_conv(h, phase_train, k=FLAGS.num_class, name='last_conv')
    logits = h
    return logits
