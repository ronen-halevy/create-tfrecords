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

import os
import json
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import argparse


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
    x_train = tf.image.resize(x_train, (size, size))

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecords_dir", type=str, default='./dataset/tfrecords',
                        help='path to tfrecords files')
    parser.add_argument("--limit", type=int, default=None,
                        help='limit on max input examples')
    parser.add_argument("--classes", type=str,
                        default='/home/ronen/PycharmProjects/shapes-dataset/dataset/shapes.names',
                        help='path to classes file')

    parser.add_argument("--batch", type=int, default=32,
                        help='batch size')

    parser.add_argument("--max_boxes", type=int, default=100,
                        help='max bounding boxes in an example image')
    args = parser.parse_args()

    tfrecords_dir = args.tfrecords_dir
    tfrecords_files = tf.io.gfile.glob(f'{tfrecords_dir}/*.tfrec')
    batch_size = args.batch
    # ds = read.get_dataset(train_filenames, batch_size)

    class_file = args.classes
    max_boxes = args.max_boxes
    LINE_NUMBER = -1
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)
    # class_table_entr = next(iter(class_table))
    pass
    files = tf.data.Dataset.list_files(tfrecords_files)
    dataset = files.map(tf.data.TFRecordDataset)

    train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrec")
    dataset = tf.data.TFRecordDataset(train_filenames)
    dataset = dataset.batch(batch_size)

    # dataset = (
    #     tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    #         .map(ReadTfrecords.parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    #         .map(ReadTfrecords.prepare_sample, num_parallel_calls=AUTOTUNE)
    #         .shuffle(batch_size * 10)
    #         .batch(batch_size)
    #
    # )


    gg= len(list(dataset))
    dataset_batch = next(iter(dataset))
    print(dataset_batch)
    dataset_batch = next(iter(dataset))

    dataset = dataset.map(lambda tfrecord: parse_tfrecord_fn(tfrecord, class_table, max_boxes, size=416))
    # gg= len(list(dataset))

    # dataset = dataset.batch(batch_size)
    gg= len(list(dataset))

    xx = dataset.take(1)

    for ww in xx:
        pass
    dataset2 = next(iter(dataset))
    for dd in dataset2:
        pass

    for dd in dataset:
        pass
    ss = dataset2[0]
    # for x, y in dataset2:
    #     pass


    # dataset = dataset.shuffle(batch_size * 10)\
    #     .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # gg = len(list(dataset))
    dataset_batch = next(iter(dataset_batch))
    # gg= len(list(dataset))
    # for x, y in dataset_batch:
    #     x, y


    dd = dataset.take(1)
    # xx,yy = dd[0], dd[1]
    pass

if __name__ == '__main__':
    main()
    # raw_dataset = tf.data.TFRecordDataset(train_filenames)#"dataset/tfrecords/file_00-15.tfrec")
    # parsed_dataset = raw_dataset.map(ReadTfrecords.parse_tfrecord_fn)
    # print(parsed_dataset)
    # for features in parsed_dataset:
    #     for key in features.keys():
    #         if key != "image/encoded":
    #             print(f"{key}: {features[key]}")
    #
    #     print(f"Image shape: {features['image/encoded'].shape}")
    #     plt.figure(figsize=(7, 7))
    #     plt.imshow(features["image/encoded"].numpy())
    #     plt.show()

# def main(_argv):
#     import tensorflow as tf
#     import pydevd;
#
#     from absl import app, flags, logging
#     from absl.flags import FLAGS
#     # flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
#
#     # haven't tested whether these are needed or not
#     tf.data.experimental.enable_debug_mode()
#     def func(x):
#         import pydevd;
#         if not FLAGS.classes:
#             pydevd.settrace(suspend=False)
#         x = x + 1  # BREAKPOINT HERE
#         tf.print("sdfsfsfsdfsfsd")
#         return x
#
#
#     dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
#     dataset = dataset.map(func)
#     for item in dataset:
#         print(item)
#     exit(0)
#
#
#
#
#     physical_devices = tf.config.experimental.list_physical_devices('GPU')
#
#     # Setup
#     if FLAGS.multi_gpu:
#         for physical_device in physical_devices:
#             tf.config.experimental.set_memory_growth(physical_device, True)
#
#         strategy = tf.distribute.MirroredStrategy()
#         print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#         BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
#         FLAGS.batch_size = BATCH_SIZE
#
#         with strategy.scope():
#             model, optimizer, loss, anchors, anchor_masks = setup_model()
#     else:
#         model, optimizer, loss, anchors, anchor_masks = setup_model()
#
#     tf.data.experimental.enable_debug_mode()
#
#     if FLAGS.dataset:
#         train_dataset = dataset.load_tfrecord_dataset(
#             FLAGS.dataset, FLAGS.classes, FLAGS.size)
#     else:
#         train_dataset = dataset.load_fake_dataset()
#     train_dataset = train_dataset.shuffle(buffer_size=512)
#     train_dataset = train_dataset.batch(FLAGS.batch_size)
#     train_dataset = train_dataset.map(lambda x, y: (
#         dataset.transform_images(x, FLAGS.size),
#         dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
#     train_dataset = train_dataset.prefetch(
#         buffer_size=tf.data.experimental.AUTOTUNE)
#
#     if FLAGS.val_dataset:
#         val_dataset = dataset.load_tfrecord_dataset(
#             FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
#     else:
#         val_dataset = dataset.load_fake_dataset()
#     val_dataset = val_dataset.batch(FLAGS.batch_size)
#     val_dataset = val_dataset.map(lambda x, y: (
#         dataset.transform_images(x, FLAGS.size),
#         dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
#
#     if FLAGS.mode == 'eager_tf':
#         # Eager mode is great for debugging
#         # Non eager graph mode is recommended for real training
#         avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
#         avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
#
#         for epoch in range(1, FLAGS.epochs + 1):
#             for batch, (images, labels) in enumerate(train_dataset):
#                 with tf.GradientTape() as tape:
#                     outputs = model(images, training=True)
#                     regularization_loss = tf.reduce_sum(model.losses)
#                     pred_loss = []
#                     for output, label, loss_fn in zip(outputs, labels, loss):
#                         pred_loss.append(loss_fn(label, output))
#                     total_loss = tf.reduce_sum(pred_loss) + regularization_loss
#
#                 grads = tape.gradient(total_loss, model.trainable_variables)
#                 optimizer.apply_gradients(
#                     zip(grads, model.trainable_variables))
#
#                 logging.info("{}_train_{}, {}, {}".format(
#                     epoch, batch, total_loss.numpy(),
#                     list(map(lambda x: np.sum(x.numpy()), pred_loss))))
#                 avg_loss.update_state(total_loss)
#
#             for batch, (images, labels) in enumerate(val_dataset):
#                 outputs = model(images)
#                 regularization_loss = tf.reduce_sum(model.losses)
#                 pred_loss = []
#                 for output, label, loss_fn in zip(outputs, labels, loss):
#                     pred_loss.append(loss_fn(label, output))
#                 total_loss = tf.reduce_sum(pred_loss) + regularization_loss
#
#                 logging.info("{}_val_{}, {}, {}".format(
#                     epoch, batch, total_loss.numpy(),
#                     list(map(lambda x: np.sum(x.numpy()), pred_loss))))
#                 avg_val_loss.update_state(total_loss)
#
#             logging.info("{}, train: {}, val: {}".format(
#                 epoch,
#                 avg_loss.result().numpy(),
#                 avg_val_loss.result().numpy()))
#
#             avg_loss.reset_states()
#             avg_val_loss.reset_states()
#             model.save_weights(
#                 'checkpoints/yolov3_train_{}.tf'.format(epoch))
#     else:
#
#         callbacks = [
#             ReduceLROnPlateau(verbose=1),
#             EarlyStopping(patience=3, verbose=1),
#             ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
#                             verbose=1, save_weights_only=True),
#             TensorBoard(log_dir='logs')
#         ]
#
#         start_time = time.time()
#         history = model.fit(train_dataset,
#                             epochs=FLAGS.epochs,
#                             callbacks=callbacks,
#                             validation_data=val_dataset)
#         end_time = time.time() - start_time
#         print(f'Total Training Time: {end_time}')
#
#
# if __name__ == '__main__':
#     import tensorflow as tf
#     import pydevd;
#
#
#     y_true_out = tf.zeros(
#         (4, 416, 416, 3, 6))
#     indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
#     updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
#
#     indexes = indexes.write(
#         0, [0, 0, 0, 0])
#
#     updates = updates.write(
#         0, [0, 0, 0, 0, 1, 0])
#
#     x = tf.tensor_scatter_nd_update(
#         y_true_out, indexes.stack(), updates.stack())
#
#     from absl import app, flags, logging
#     from absl.flags import FLAGS
#     # flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
#
#     # haven't tested whether these are needed or not
#     tf.data.experimental.enable_debug_mode()
#     # tf.config.run_functions_eagerly(True)
#
#
#
#     try:
#         app.run(main)
#     except SystemExit:
#         pass
#
