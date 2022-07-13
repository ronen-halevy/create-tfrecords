#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_tfrecord.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description :
#
# ================================================================

import os
import glob
import json
import tensorflow as tf
import numpy as np
import argparse


class ExampleProtos:
    @staticmethod
    def image_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    @staticmethod
    def bytes_feature_list(value):
        value = [x.encode('utf8') for x in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_feature_list(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature_list(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, example):
    """

    :param image:
    :type image:
    :param example:
    :type example:
    :return:
    :rtype:
    """
    boxes = np.reshape(example['bboxes'], -1)
    label = [entry for entry in example['labels']]

    feature = {
        'image/encoded': ExampleProtos.image_feature(image),
        'image/object/bbox/xmin': ExampleProtos.float_feature_list(boxes[0::4].tolist()),
        'image/object/bbox/ymin': ExampleProtos.float_feature_list(boxes[1::4].tolist()),
        'image/object/bbox/xmax': ExampleProtos.float_feature_list(boxes[2::4].tolist()),
        'image/object/bbox/ymax': ExampleProtos.float_feature_list(boxes[3::4].tolist()),
        'image/object/class/text': ExampleProtos.bytes_feature_list(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tfrecords(input_annotation_file, images_dir, tfrecords_out_dir, tfrec_file_size, train_split, val_split, examples_limit=None):
    """

    :param input_annotation_file:
    :type input_annotation_file:
    :param images_dir:
    :type images_dir:
    :param tfrecords_out_dir:
    :type tfrecords_out_dir:
    :param tfrec_file_size:
    :type tfrec_file_size:
    :param examples_limit:
    :type examples_limit:
    :return:
    :rtype:
    """

    train_dir = f'{tfrecords_out_dir}/train'
    val_dir = f'{tfrecords_out_dir}/val'
    test_dir = f'{tfrecords_out_dir}/test'

    for out_dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            to_del_files = glob.glob(f'{out_dir}/*.tfrec')
            [os.remove(f) for f in to_del_files]


    with open(input_annotation_file, 'r') as f:
        annotations = json.load(f)['annotations']

    num_examples = min(len(annotations), examples_limit or float('inf'))
    train_size = int(train_split * num_examples)
    val_size = int(val_split * num_examples)
    test_size = num_examples - train_size - val_size

    start_record = 0
    for split_size, out_dir in zip([train_size, val_size, test_size], [train_dir, val_dir, test_dir]):
        num_samples_in_tfrecord = min(tfrec_file_size, split_size)
        num_tfrecords = split_size // num_samples_in_tfrecord
        if split_size % num_samples_in_tfrecord:
            num_tfrecords += 1
        # split_annotations = annotations[start_record: min(start_record + num_tfrecords, len(annotations)-1)]
        print(f'Starting! \nCreating {split_size} examples in {num_tfrecords} tfrecord files.')
        print(f'Output dir: {tfrecords_out_dir}')

        for tfrec_num in range(num_tfrecords):
            samples = annotations[((start_record+tfrec_num) * num_samples_in_tfrecord): ((start_record+tfrec_num + 1) * num_samples_in_tfrecord)]

            with tf.io.TFRecordWriter(
                    f'{out_dir}/file_{tfrec_num:02}_{len(samples)}.tfrec'
            ) as writer:
                for sample in samples:
                    image_path = images_dir + sample['image_filename']
                    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                    example = create_example(image, sample)
                    writer.write(example.SerializeToString())
        start_record = start_record + num_tfrecords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_limit", type=int, default=None,
                        help="limit to num of processed examples")

    parser.add_argument("--out_dir", type=str, default='./dataset/tfrecords',
                        help='output dir')

    parser.add_argument("--images_dir", type=str, default='/home/ronen/PycharmProjects/shapes-dataset/dataset/images/',
                        help='input base_dir')

    parser.add_argument("--in_annotations", type=str,
                        default='/home/ronen/PycharmProjects/shapes-dataset/dataset/annotations/annotations.json',
                        help='input annotations meta data')

    parser.add_argument("--tfrec_file_size", type=int,
                        default=4096,
                        help='number of examples in a tfrec file')

    parser.add_argument("--train_split", type=float,
                        default=0.7,
                        help='train_split fraction')

    parser.add_argument("--val_split", type=float,
                        default=0.2,
                        help='val_split fraction')

    args = parser.parse_args()
    in_annotations = args.in_annotations
    out_dir = args.out_dir
    images_dir = args.images_dir
    tfrec_file_size = args.tfrec_file_size
    examples_limit = args.examples_limit
    train_split = args.train_split
    val_split = args.val_split
    test_split = max(0, 1-(train_split+val_split))
    create_tfrecords(in_annotations, images_dir, out_dir, tfrec_file_size, train_split, val_split, examples_limit)
    print('Done!')


if __name__ == '__main__':
    main()
