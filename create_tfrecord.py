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
    boxes = np.reshape(example['bboxes'], -1)
    label = [entry['label'] for entry in example['objects']]

    feature = {
        'image/encoded': ExampleProtos.image_feature(image),
        'image/object/bbox/xmin': ExampleProtos.float_feature_list(boxes[0::4].tolist()),
        'image/object/bbox/ymin': ExampleProtos.float_feature_list(boxes[1::4].tolist()),
        'image/object/bbox/xmax': ExampleProtos.float_feature_list(boxes[2::4].tolist()),
        'image/object/bbox/ymax': ExampleProtos.float_feature_list(boxes[3::4].tolist()),
        'image/object/class/label': ExampleProtos.bytes_feature_list(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tfrecords(input_annotation_file, images_dir, tfrecords_out_dir, examples_in_file, examples_limit=None):
    with open(input_annotation_file, 'r') as f:
        annotations = json.load(f)['annotations']

    num_examples = min(len(annotations), examples_limit or float('inf'))
    num_samples_in_tfrecord = min(examples_in_file, num_examples)
    num_tfrecords = num_examples // num_samples_in_tfrecord
    if num_examples % num_samples_in_tfrecord:
        num_tfrecords += 1

    if not os.path.exists(tfrecords_out_dir):
        os.makedirs(tfrecords_out_dir)
    else:
        to_del_files = glob.glob(f'{tfrecords_out_dir}/*.tfrec')
        [os.remove(f) for f in to_del_files]

    print(f'Starting! \nCreating {num_examples} examples in {num_tfrecords} tfrecord files.')
    print(f'Output dir: {tfrecords_out_dir}')

    for tfrec_num in range(num_tfrecords):
        samples = annotations[(tfrec_num * num_samples_in_tfrecord): ((tfrec_num + 1) * num_samples_in_tfrecord)]

        with tf.io.TFRecordWriter(
                tfrecords_out_dir + '/file_%.2i-%i.tfrec' % (tfrec_num, len(samples))
        ) as writer:
            for sample in samples:
                image_path = images_dir + sample['image_filename']
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, sample)
                writer.write(example.SerializeToString())

    print('Done!')


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

    parser.add_argument("--examples_in_tfrec", type=str,
                        default=4096,
                        help='number of examples in a tfrec file')

    args = parser.parse_args()
    in_annotations = args.in_annotations
    out_dir = args.out_dir
    images_dir = args.images_dir
    examples_in_file = args.examples_in_file
    examples_limit = args.examples_limit
    create_tfrecords(in_annotations, images_dir, out_dir, examples_in_file, examples_limit)


if __name__ == '__main__':
    main()
