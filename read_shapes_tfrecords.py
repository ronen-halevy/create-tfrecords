#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : read_shapes_tfrecord.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description :
#
# ================================================================


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import argparse

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont


def parse_tfrecord_fn(tfrecord, class_table, max_boxes, size):
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


def draw_bounding_box(image, boxes, color, thickness=3):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)
    return image


def draw_text_on_bounding_box(image, ymin, xmin, color, display_str_list=(), font_size=30):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  font_size)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    text_margin_factor = 0.05

    text_widths, text_heights = zip(*[font.getsize(display_str) for display_str in display_str_list])
    text_margins = np.ceil(text_margin_factor * np.array(text_heights))
    text_bottoms = ymin * (ymin > text_heights) + (ymin + text_heights) * (ymin <= text_heights)

    for idx, (display_str, xmint, text_bottom, text_width, text_height, text_margin) in enumerate(
            zip(display_str_list, xmin, text_bottoms, text_widths, text_heights, text_margins)):
        text_width, text_height = font.getsize(display_str)
        text_margin = np.ceil(text_margin_factor * text_height)

        draw.rectangle(((xmint, text_bottom - text_height - 2 * text_margin),
                        (xmint + text_width + text_margin, text_bottom)),
                       fill=color)

        draw.text((xmint + text_margin, text_bottom - text_height - text_margin),
                  display_str,
                  fill="black",
                  font=font)
    return image


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

    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        filename=class_file, key_dtype=tf.string, key_index=0, value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER, delimiter="\n"), default_value=-1)

    files = tf.data.Dataset.list_files(f"{tfrecords_dir}/*.tfrec")

    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda tfrecord: parse_tfrecord_fn(tfrecord, class_table, max_boxes, size=416))

    data = dataset.take(1)
    image, y = next(iter(data))

    y = y[y[..., 2].numpy() != 0]  # remove padding
    image_pil = Image.fromarray(np.uint8(image.numpy() * 255))
    annotated_bbox_image = draw_bounding_box(image_pil, y[..., 0:4], color=(255, 255, 0),
                                             thickness=3)

    colors = list(ImageColor.colormap.values())
    color = colors[0]
    class_text = np.loadtxt(class_file, dtype=str)

    classes = class_text[y[..., 4].numpy().astype(int)]
    annotated_text_image = draw_text_on_bounding_box(annotated_bbox_image, y[..., 1].numpy(), y[..., 0].numpy(), color,
                                                     classes, font_size=15)

    plt.imshow(annotated_text_image)
    plt.show()


if __name__ == '__main__':
    main()