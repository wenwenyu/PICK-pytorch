#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""This file"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os
import json
import random
import shutil
import urllib.request
from typing import Dict

import cv2
from PIL import Image
import numpy as np


__author__ = "TengQi Ye"
__copyright__ = "Copyright 2020-2020"
__credits__ = ["TengQi Ye"]
__license__ = ""
__version__ = "0.0.2"
__maintainer__ = "TengQi Ye"
__email__ = "yetengqi@gmail.com"
__status__ = "Research"


def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')

    return parser.parse_args()


def get_image_shape(img_path):
    image = cv2.imread(img_path)
    return image.shape


def save_json_file(data, json_filename, force=False):
    """Save json file."""
    if not force:
        assert not os.path.exists(json_filename), f'{json_filename} already exists!'
    # Make directory if not exists.
    mkdir(os.path.dirname(json_filename))

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent='')
    print(f'Successfully save {json_filename} ...')


def load_json_file(json_filename, verbose=False):
    """Load json file given filename."""
    assert os.path.exists(json_filename), f'{json_filename} does not exists!'

    if verbose:
        print(f'Loading {json_filename} ...')
    with open(json_filename, 'r') as f:
        return json.load(f)


def save_img_with_bboxes(img_name, target_img_name, bounding_boxes=None,
                         color_groups=None, labels=None, thickness_list=None):
    # todo: refinement
    if color_groups is not None:
        # The same item in color_groups share the same color.
        color_dict = {}
        for group in color_groups:
            if group not in color_dict:
                color_dict[group] = tuple(random.sample(range(255), k=3))

        # print(f'Plot with {len(color_dict)} groups.')

    # Plotting.
    if type(img_name) == str:
        image = cv2.imread(img_name, 1)
    else:
        image = img_name

    for idx, box in enumerate(bounding_boxes):
        color = None if color_groups is None else color_dict[color_groups[idx]]
        label = None if labels is None else labels[idx]
        thickness = None if thickness_list is None else thickness_list[idx]
        draw_bbox_on_image(image, box, color=color,
                           thickness=thickness, text=label)

    if os.path.splitext(target_img_name)[1] == '':  # Filename without extension.
        target_img_name += '.jpg'

    target_dir = os.path.dirname(target_img_name)
    # mkdir(os.path.dirname(target_img_name)) # todo: it may cause threading problem.
    assert os.path.exists(target_dir), f'{target_dir} does not exists!'
    if not cv2.imwrite(target_img_name, image):
        print(f'save_img_with_bboxes-> Fail to save {target_img_name} ...')


def plot_img_with_bboxes(img_name, bounding_boxes, labels=None):
    image = cv2.imread(img_name, 1)
    for idx, box in enumerate(bounding_boxes):
        label = None if labels is None else labels[idx]
        draw_bbox_on_image(image, box, text=label)

    Image.fromarray(image).show(title=img_name)


def get_json(object):
    return json.loads(
        json.dumps(object, default=lambda o: getattr(o, '__dict__', str(o)))
    )


def copy_or_move_file(src, dst, mode='copy', verbose=False):
    assert mode in ('copy', 'move'), 'Mode can only be copy or move.'
    if verbose: assert not os.path.exists(dst), f'{mode}->{dst} already exists!'
    if mode == 'copy':
        shutil.copy(src, dst)
    else:
        shutil.move(src, dst)

    if verbose:
        print(f'{mode} {src} to {dst} ...')


def plot_img_with_box(img_name, box):
    # todo: refinefe
    image = cv2.imread(img_name, 1)
    draw_bbox_on_image(image, box)

    # Resize in case too big.
    # cv2.imshow('Bounding boxes', image)
    # cv2.waitKey(0)
    Image.fromarray(image).show(title=img_name)


def draw_bbox_on_image(image, bbox, color=None, thickness=None, text=None):
    """
    Draw a bounding box on an image. (memory)

    :param image: numpy array or str (path to image).
    :param bbox: [x0, y0, w, h]
    :return:
    """
    if color is None:
        color = (255, 0, 0)

    if thickness is None:
        thickness = 2

    # Read image if it is a path.
    if type(image) == str:
        assert os.path.exists(image), f'{image} does not exists!'
        image = cv2.imread(image)

    # Unpack coordinates.
    x, y, w, h = [int(b) for b in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    # Draw text.
    if text is not None:
        cv2.putText(image, str(text), (x, y), 0, 3, color, 2)


def mkdir(folder, verbose=False, force=False):
    ext = os.path.exists(folder)
    if not ext:
        os.makedirs(folder)
        if verbose: print(f'Successfully create {folder} ...')
    else:
        if verbose: print(f'{folder} already exists!')
        if force:
            shutil.rmtree(folder)
            print(f'Recreating {folder} !!!')
            os.makedirs(folder)


def IoU(box1, box2):
    """
    Compute the IoU of two bounding boxes.
    :param box1: [x, y, w, h]
    :param box2: [x, y, w, h]
    :return:
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xl1, yl1, xr1, yr1 = x1, y1, x1 + w1, y1 + h1
    xl2, yl2, xr2, yr2 = x2, y2, x2 + w2, y2 + h2
    overlap_w = max(min(xr1, xr2) - max(xl1, xl2), 0)
    overlap_h = max(min(yr1, yr2) - max(yl1, yl2), 0)

    return overlap_w * overlap_h / (w1 * h1 + w2 * h2 - overlap_w * overlap_h)


def Intersection(box1, box2):
    """
    Compute the IoU of two bounding boxes.
    :param box1: [x, y, w, h]
    :param box2: [x, y, w, h]
    :return:
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xl1, yl1, xr1, yr1 = x1, y1, x1 + w1, y1 + h1
    xl2, yl2, xr2, yr2 = x2, y2, x2 + w2, y2 + h2
    overlap_w = max(min(xr1, xr2) - max(xl1, xl2), 0)
    overlap_h = max(min(yr1, yr2) - max(yl1, yl2), 0)

    return overlap_w * overlap_h


def split_line_on_comma(line):
    line = line.strip().strip(',')
    tokens = line.split(',')
    return tokens


def draw_bboxes_with_ann(src_img_name, ann_filename, des_img_name,
                         label_to_color: Dict):
    # ann_filename: entity annotation file.

    assert os.path.exists(src_img_name), f'{src_img_name} does not exists!'
    image = cv2.imread(src_img_name, cv2.IMREAD_COLOR)

    # Plot polynomials.
    if not os.path.exists(ann_filename):
        return

    with open(ann_filename, 'r') as ann_file:
        lines = ann_file.readlines()

    for idx, line in enumerate(lines):
        # Process the line.
        tokens = split_line_on_comma(line)
        index, label = tokens[0], tokens[-1]
        label = label.strip()  # Remove spaces in the term.
        points = np.asarray([int(float(token)) for token in tokens[1:9]])
        points = np.reshape(points, (-1, 2))

        plot_box_with_label(image, points, f'{index}: {label}', color=label_to_color[label])

    # Save the image.
    if not cv2.imwrite(des_img_name, image):
        print(f'Fail to save the {des_img_name}')


def plot_box_with_label(image, points, text, color=(0, 0, 0)):
    # points = [[x1, y1], [x2, y2], ...]

    # Plot polynomials.
    image = cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

    # Plot text.
    xs, ys = zip(*points)
    x, y = min(xs), min(ys)

    # In the future, the text may exceed the boundary.
    # https://stackoverflow.com/questions/56660241/how-to-wrap-text-in-opencv-when-i-print-it-on-an-image-and-it-exceeds-the-frame
    # There is a bug here causing long long lines.
    cv2.putText(image, text, (x, y), 0, 0.5, color, 2)