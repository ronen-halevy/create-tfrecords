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
import yaml
import json
import tensorflow as tf
import numpy as np
import argparse
import math

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


def create_example(image, annotations, class_names, sparse_to_dense_category_id, width, height):
    """

    :param image:
    :type image:
    :param example:
    :type example:
    :return:
    :rtype:
    """
    bboxes = []
    categories = []
    for annot in annotations:
        bboxes.append(annot['bbox'])
        categories.append(sparse_to_dense_category_id[annot['category_id']]) # trnaslated to dense category id (no holess in numbering sequence)

    scale = np.array([width, height, width, height])
    bboxes = np.array(bboxes)/scale
    bboxes = np.reshape(bboxes, -1)
    class_names = np.array(class_names)
    classes = class_names[categories]
    bbox_width = np.array(bboxes[2::4]).astype(float)
    bbox_height = np.array(bboxes[3::4]).astype(float)

    feature = {
        'image/encoded': ExampleProtos.image_feature(image),
        'image/object/bbox/xmin': ExampleProtos.float_feature_list(bboxes[0::4].tolist()),
        'image/object/bbox/ymin': ExampleProtos.float_feature_list(bboxes[1::4].tolist()),
        'image/object/bbox/xmax': ExampleProtos.float_feature_list((bboxes[0::4] + (bbox_width)).tolist()),
        'image/object/bbox/ymax': ExampleProtos.float_feature_list((bboxes[1::4] + (bbox_height)).tolist()),
        'image/object/class/text': ExampleProtos.bytes_feature_list(classes),
        'image/object/class/id': ExampleProtos.float_feature_list(categories),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def calc_num_images_in_tfrecord_file(images_list, images_dir, optimal_file_size):
    acc_images_size = 0
    for image_entry in images_list:
        image_path = images_dir + image_entry['file_name']
        acc_images_size += os.path.getsize(image_path)
    avg_image_file_size = acc_images_size / len(images_list)
    num_images_in_tfrecord_file = int(optimal_file_size // avg_image_file_size)
    return num_images_in_tfrecord_file



def write_tfrecord_file(tfrecord_idx, image_entries, annotations_list, images_dir, out_dir, class_names, sparse_to_dense_category_id):

    with tf.io.TFRecordWriter(
            f'{out_dir}/file_{tfrecord_idx:02}_{len(image_entries)}.tfrec'
    ) as writer:
        for image_entry in image_entries:
            annots = [anno for anno in annotations_list if anno['image_id'] == image_entry['id']]
            image_path = images_dir + image_entry['file_name']
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            im_width = image_entry['width']
            im_height = image_entry['height']
            example = create_example(image, annots, class_names, sparse_to_dense_category_id, im_width, im_height)
            writer.write(example.SerializeToString())

def create_tfrecords(input_annotations_file,
                     images_dir,
                     tfrecords_out_dir,
                     optimal_file_size,
                     train_split,
                     val_split,
                     classes_name_file,
                     examples_limit=None):
    """

    :param input_annotations_file:
    :type input_annotations_file:
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
    class_names = [c.strip() for c in open(classes_name_file).readlines()]


    train_dir = f'{tfrecords_out_dir}/train'
    val_dir = f'{tfrecords_out_dir}/val'
    test_dir = f'{tfrecords_out_dir}/test'

    for out_dir in [train_dir, val_dir, test_dir]:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            to_del_files = glob.glob(f'{out_dir}/*.tfrec')
            [os.remove(f) for f in to_del_files]

    with open(input_annotations_file, 'r') as f:
        annotations = json.load(f)

    annotations_list = annotations['annotations']
    num_examples = min(len(annotations['images']), examples_limit or float('inf'))
    images_list = annotations['images'][0:num_examples]
    categories_list = annotations['categories']
    sparse_to_dense_category_id = {category['id']: index for index, category in enumerate(categories_list)}

    # split dataset:
    train_images_list, remainder = np.split(images_list, [int(train_split * len(images_list))])
    val_images_list, test_images_list = np.split(remainder, [int((val_split) * len(images_list))])

    num_images_in_tfrecord_file = calc_num_images_in_tfrecord_file(images_list, images_dir, optimal_file_size)

    for split_images_list, out_dir in zip([train_images_list, val_images_list, test_images_list], [train_dir, val_dir, test_dir]):
        # find number of tfrec files needed:
        num_tfrecords_files = int(math.ceil(len(split_images_list) / num_images_in_tfrecord_file))
        print(f'Starting! \nCreating {len(split_images_list)} examples in {num_tfrecords_files} tfrecord files.')
        print(f'Output dir: {tfrecords_out_dir}')
        start_record = 0
        # calc mum of images per tfrec:
        tfrec_files_sizes = np.tile([len(split_images_list) / num_tfrecords_files], num_tfrecords_files).astype(int)
        # spill entries remainder to last tfrecord file:
        tfrec_files_sizes[-1] = tfrec_files_sizes[-1] + (len(split_images_list) -  num_tfrecords_files *sum(tfrec_files_sizes))
        for tfrecord_idx, tfrec_file_size in enumerate(tfrec_files_sizes):
            write_tfrecord_file(tfrecord_idx, split_images_list[start_record: tfrec_file_size], annotations_list, images_dir, out_dir, class_names, sparse_to_dense_category_id)
            start_record = start_record + tfrec_file_size

    # save a copy of class names file here:
    class_name_path = f'{tfrecords_out_dir}/class.names'
    with open(class_name_path, 'w') as f:
        for class_name in class_names:
            f.write("%s\n" % class_name)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str,
                        default='config/config.yaml',
                        help='config file')

    args = parser.parse_args()
    config_file = args.config

    with open(config_file, 'r') as stream:
        configs = yaml.safe_load(stream)

    create_tfrecords(**configs)

    print('Done!')


if __name__ == '__main__':
    main()
