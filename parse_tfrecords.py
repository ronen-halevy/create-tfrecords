# ================================================================
# ! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : parse_tfrecord.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description : Parse tfrecords read from filess and return a dataset . The tfrecord entries structure consists of
#   an image, with bounding boxes and class labels metadatas.


import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
import yaml

from render_utils import render_dataset_examples

def parse_tfrecord_fn(record, image_size, max_boxes, class_table=None):
    """

    :param record: A record read from TfRecord files
    :type record:
    :param class_table: A table to map string class labels to number representation. If missing, dataset's class
    entries are ignored.
    :type class_table:
    :param max_boxes: Pad number of boxes to max_boxes, to achieve uniform size. Caution! Fails if number of boxes in
    any entry exeeds > max_boxes
    :type max_boxes:
    :param image_size: Assumed square. Resizing, otherwise image shapes values are None.
    :type image_size:
    :return: Transformed dataset
    :rtype:
    """
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/class/text": tf.io.VarLenFeature(tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(record, feature_description)

    x_train = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (image_size, image_size)) / 255

    labels = tf.sparse.to_dense(
        example.get('image/object/class/text', ','), default_value='')
    if class_table:
        labels = tf.cast(class_table.lookup(labels), tf.float32)
        y_train = tf.stack([tf.sparse.to_dense(example['image/object/bbox/xmin']),
                            tf.sparse.to_dense(example['image/object/bbox/ymin']),
                            tf.sparse.to_dense(example['image/object/bbox/xmax']),
                            tf.sparse.to_dense(example['image/object/bbox/ymax']),
                            labels], axis=1)
    else:
        labels = tf.cast(tf.fill([x_train.shape[0], 1], 1). tf.float32)
        y_train = tf.stack([tf.sparse.to_dense(example['image/object/bbox/xmin']),
                            tf.sparse.to_dense(example['image/object/bbox/ymin']),
                            tf.sparse.to_dense(example['image/object/bbox/xmax']),
                            tf.sparse.to_dense(example['image/object/bbox/ymax']),
                            labels
                            ], axis=1)

    paddings = [[0, max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def parse_tfrecords(tfrecords_dir, image_size, max_boxes, class_file=None):
    """
    :param tfrecords_dir:
    :type tfrecords_dir:
    :param max_boxes:
    :type max_boxes:
    :param class_file: If none, class labels will not be assigned to integers
    :type class_file:
    :return:
    :rtype:
    """

    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        filename=class_file, key_dtype=tf.string, key_index=0, value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER, delimiter="\n"), default_value=-1) if class_file else None
    files = tf.data.Dataset.list_files(f"{tfrecords_dir}/*.tfrec")

    dataset = files.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.map(lambda record: parse_tfrecord_fn(record, image_size, max_boxes, class_table))
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str,
                        default='config/parser_config.yaml',
                        help='config file')

    args = parser.parse_args()
    config_file = args.config

    with open(config_file, 'r') as stream:
        configs = yaml.safe_load(stream)
    dataset = parse_tfrecords(**configs)
    class_file = configs['class_file']
    annotated_text_image = render_dataset_examples(dataset, class_file)
    plt.imshow(annotated_text_image)
    plt.show()


if __name__ == '__main__':
    main()
