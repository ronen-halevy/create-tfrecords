#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : read_tfrecord.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description : Reads a tfrecord file . The tfrecord entry is expected to
#   hold an image, with bounding box and a class label metadata
#   main() call rendering method which plots example dataset entries.
# ================================================================


import tensorflow as tf
from matplotlib import pyplot as plt
import argparse

from render_dataset import render_dataset_examples


def parse_tfrecord_fn(tfrecord, class_table, max_boxes, size):
    """

    :param tfrecord:
    :type tfrecord:
    :param class_table:
    :type class_table:
    :param max_boxes:
    :type max_boxes:
    :param size:
    :type size:
    :return:
    :rtype:
    """
    feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/class/label": tf.io.VarLenFeature(tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(tfrecord, feature_description)

    x_train = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size)) / 255

    labels = tf.sparse.to_dense(
        example.get('image/object/class/label', ','), default_value='')
    labels = tf.cast(class_table.lookup(labels), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(example['image/object/bbox/xmin']),
                        tf.sparse.to_dense(example['image/object/bbox/ymin']),
                        tf.sparse.to_dense(example['image/object/bbox/xmax']),
                        tf.sparse.to_dense(example['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def read_dataset(class_file, tfrecords_dir, max_boxes):
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        filename=class_file, key_dtype=tf.string, key_index=0, value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER, delimiter="\n"), default_value=-1)

    files = tf.data.Dataset.list_files(f"{tfrecords_dir}/*.tfrec")

    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda tfrecord: parse_tfrecord_fn(tfrecord, class_table, max_boxes, size=416))
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecords_dir", type=str, default='./dataset/tfrecords',
                        help='path to tfrecords files')
    parser.add_argument("--limit", type=int, default=None,
                        help='limit on max input examples')
    parser.add_argument("--classes", type=str,
                        default='/home/ronen/PycharmProjects/shapes-dataset/dataset/class.names',
                        help='path to classes names file needed to annotate plotted objects')

    parser.add_argument("--max_boxes", type=int, default=100,
                        help='max bounding boxes in an example image')
    args = parser.parse_args()

    tfrecords_dir = args.tfrecords_dir

    class_file = args.classes
    max_boxes = args.max_boxes
    dataset = read_dataset(class_file, tfrecords_dir, max_boxes)
    annotated_text_image = render_dataset_examples(dataset, class_file)
    plt.imshow(annotated_text_image)
    plt.show()


if __name__ == '__main__':
    main()
