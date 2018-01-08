""" tfrecord.py
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import make_dirs


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_to_tfrecord(pairs, outdir, name):
    make_dirs(outdir)

    writer = tf.python_io.TFRecordWriter(os.path.join(outdir, name))
    print('Writing', name)
    for image_path, label_path in pairs:
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))
        height = image.shape[0]
        width = image.shape[1]

        image_raw = image.tostring()
        label_raw = label.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(image_raw),
            'label_raw': _bytes_feature(label_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
