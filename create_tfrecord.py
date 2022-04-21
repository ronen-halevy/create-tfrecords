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
from absl import app, flags
from absl.flags import FLAGS


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


def create_tfrecords(input_annotation_file, tfrecords_out_dir, examples_limit=None):
    with open(input_annotation_file, 'r') as f:
        annotations = json.load(f)['annotations']

    num_examples = min(len(annotations), examples_limit or float('inf'))
    num_samples_in_tfrecord = min(4096, num_examples)
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
                image_path = sample['image_path']
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, sample)
                writer.write(example.SerializeToString())

    print('Done!')


def main():
    flags.DEFINE_string('outdir', 'dataset/tfrecords', 'path to tfrecords outfiles')
    flags.DEFINE_string('annotations', 'dataset/annotations/annotations.json', 'path to anonotations infile')
    flags.DEFINE_integer('limit', None, 'limit on max input examples')

    tfrecords_out_dir = FLAGS.outdir
    input_annotation_file = FLAGS.annotations
    examples_limit = FLAGS.limit
    create_tfrecords(input_annotation_file, tfrecords_out_dir, examples_limit)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
