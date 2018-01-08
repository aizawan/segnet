""" inputs.py
"""


import tensorflow as tf


def replace_ignore_label(label, ignore_label=255):
    cond = tf.equal(label, ignore_label)
    cond_true = tf.constant(-1, shape=label.get_shape())
    cond_false = label
    replaced_label = tf.where(cond, cond_true, cond_false)
    return replaced_label


# https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/augmentation.py
def randomly_scale(image, label, output_shape,
                   min_scale=0.9, max_scale=1.1, ignore_label=255):
    bx = tf.expand_dims(image, 0)
    by = tf.expand_dims(label, 0)

    by = tf.to_int32(by)

    input_shape = tf.to_float(tf.shape(bx)[1:3])
    scale_shape = output_shape / input_shape

    rand_var = tf.random_uniform(shape=[1], minval=min_scale, maxval=max_scale)
    scale = tf.reduce_min(scale_shape) * rand_var

    scaled_input_shape = tf.to_int32(tf.round(input_shape * scale))

    resized_image = tf.image.resize_nearest_neighbor(bx, scaled_input_shape)
    resized_label = tf.image.resize_nearest_neighbor(by, scaled_input_shape)

    resized_image = tf.squeeze(resized_image, axis=0)
    resized_label = tf.squeeze(resized_label, axis=0)

    shifted_classes = resized_label + 1

    cropped_image = tf.image.resize_image_with_crop_or_pad(
        resized_image, output_shape[0], output_shape[1])

    cropped_label = tf.image.resize_image_with_crop_or_pad(
        shifted_classes, output_shape[0], output_shape[1])

    mask = tf.to_int32(tf.equal(cropped_label, 0)) * (ignore_label + 1)
    cropped_label = cropped_label + mask - 1

    return cropped_image, cropped_label


def scale_fixed_size(raw_image, raw_label, output_shape, ignore_label=255):
    raw_image = tf.cast(raw_image, tf.float32) / 255.
    raw_label = tf.cast(raw_label, tf.int32)
    raw_height = tf.shape(raw_image)[0]
    raw_width = tf.shape(raw_image)[1]

    image_batch = tf.expand_dims(raw_image, 0)
    label_batch = tf.expand_dims(raw_label, 0)
    raw_label_size = tf.shape(image_batch)
    raw_image_size = tf.shape(label_batch)

    input_shape = tf.to_float(raw_image_size[1:3])
    scale_shape = output_shape / input_shape
    scale = tf.reduce_min(scale_shape)
    scaled_input_shape = tf.to_int32(tf.round(input_shape * scale))

    resized_image = tf.image.resize_nearest_neighbor(
        image_batch, scaled_input_shape)
    resized_label = tf.image.resize_nearest_neighbor(
        label_batch, scaled_input_shape)

    shifted_classes = resized_label + 1

    cropped_image = tf.image.resize_image_with_crop_or_pad(
        resized_image, output_shape[0], output_shape[1])
    cropped_label = tf.image.resize_image_with_crop_or_pad(
        shifted_classes, output_shape[0], output_shape[1])

    mask = tf.to_int32(tf.equal(cropped_label, 0)) * (ignore_label + 1)
    cropped_label = cropped_label + mask - 1

    return cropped_image, resized_image, mask


def read_image_label_from_queue(image_list, label_list):
    image_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
    label_tensor = tf.convert_to_tensor(label_list, dtype=tf.string)
    queue = tf.train.slice_input_producer(
        [image_tensor, label_tensor], shuffle=False)

    image_contents = tf.read_file(queue[0])
    label_contents = tf.read_file(queue[1])

    images = tf.image.decode_png(image_contents, channels=3)
    labels = tf.image.decode_png(label_contents, channels=1)
    return images, labels


def read_and_decode(fn_queue, target_height, target_width, batch_size,
                    num_threads, min_after_dequeue, shuffle=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fn_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image = tf.reshape(image, [height, width, 3])
    label = tf.reshape(label, [height, width, 1])

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    image = image / 255.

    resized_image, resized_label = randomly_scale(
        image, label, [target_height, target_width])

    resized_label = replace_ignore_label(resized_label)

    capacity = min_after_dequeue + num_threads * batch_size

    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [resized_image, resized_label],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=num_threads,
            min_after_dequeue=min_after_dequeue)
    else:
        images, labels = tf.train.batch(
            [resized_image, resized_label],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=num_threads)

    return images, labels
