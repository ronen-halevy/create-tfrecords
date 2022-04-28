#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : render_tfrecords.py
#   Author      : ronen halevy 
#   Created date:  4/27/22
#   Description :  Render boundingBox and Text annotations overlays for object detection dataset
#
# ================================================================
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont


def draw_text_on_bounding_box(image, ymin, xmin, color, display_str_list=(), font_size=30):
    """

    :param image: Image
    :type image:
    :param ymin: Text is placed above this coordinate, unless it is near image's upper edge.
    :type ymin:
    :param xmin: Text is place to the right of this coordinate
    :type xmin:
    :param color: Text background rectangle color
    :type color: numbers or tuples
    :param display_str_list: string to display
    :type display_str_list: str
    :param font_size: Text font size
    :type font_size: int
    :return: image with text annotations
    :rtype: Image
    """
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


def draw_bounding_box(image, boxes, color, thickness=1):
    """
    :param image:
    :type image:
    :param boxes:
    :type boxes:
    :param color:
    :type color:
    :param thickness:
    :type thickness:
    :return:
    :rtype:
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)
    return image


def render_dataset_examples(dataset, class_file):
    """

    :param dataset:
    :type dataset:
    :param class_file:
    :type class_file:
    :return:
    :rtype:
    """
    data = dataset.take(1)
    image, y = next(iter(data))

    y = y[y[..., 2].numpy() != 0]  # remove padding
    image_pil = Image.fromarray(np.uint8(image.numpy() * 255))
    annotated_bbox_image = draw_bounding_box(image_pil, y[..., 0:4], color=(255, 255, 0),
                                             thickness=1)

    colors = list(ImageColor.colormap.values())
    color = colors[0]
    class_text = np.loadtxt(class_file, dtype=str)

    classes = class_text[y[..., 4].numpy().astype(int)]
    annotated_text_image = draw_text_on_bounding_box(annotated_bbox_image, y[..., 1].numpy(), y[..., 0].numpy(), color,
                                                     classes, font_size=15)
    return annotated_text_image
