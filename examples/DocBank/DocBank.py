#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os, time
import threading
from typing import List
from multiprocessing import Pool
from functools import partial

import cv2
from PIL import Image

from examples import utils


def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--des_dir', type=str)

    parser.add_argument('--src_imgs_dir', type=str)
    parser.add_argument('--src_raw_ann_dir', type=str)
    return parser.parse_args()


def xyxy_to_box_coord(x0, y0, x1, y1):
    # Convert xyxy (strings) to box coordinates (strings).
    box_coord = [x0, y0, x1, y0, x1, y1, x0, y1]

    return ','.join(box_coord)


def ann_convert(src_filename: str, des_filename: str, src_imgname: str):
    """
    Convert raw annotation files to the files with format fitting PICK.

    Line of source file: transcripts, x0, y0, x1, y1, R, G, B, font, label
    Line of destination file: line_index, box_coordinates, transcripts, label.
    src_imgname: the image name of source file.
    """
    assert os.path.isfile(src_filename), f'{src_filename} does not exists!'
    # Read lines from source file.
    with open(src_filename, 'r') as file:
        src_lines = file.readlines()

    assert os.path.isfile(src_imgname), f'{src_imgname} does not exists!'

    if len(src_lines) == 0:
        print(f'Fail to process {src_filename} ...')
        return False  # Skip the file.

    # We need the intermediate reuslt.
    split_token_list = []
    for line_idx, line in enumerate(src_lines):
        split_tokens = line.strip().split('\t')
        if len(split_tokens) == 10:
            transcripts, x0, y0, x1, y1, _, _, _, _, label = split_tokens
        else:
            # We assume the default transcripts is the space.
            transcripts = ' '
            x0, y0, x1, y1, _, _, _, _, label = split_tokens

        x0, y0, x1, y1 = adjust_box(src_imgname, x0, y0, x1, y1)
        split_token_list.append([transcripts, x0, y0, x1, y1, label])

    # Write format.
    des_lines = []
    for line_idx, line in enumerate(split_token_list):
        transcripts, x0, y0, x1, y1, label = line
        des_lines.append(f'{line_idx},{xyxy_to_box_coord(x0, y0, x1, y1)},{transcripts},{label}\n')

    with open(des_filename, 'w') as file:
        file.writelines(des_lines)

    return True


def adjust_box(img_filename, x0, y0, x1, y1):
    """
    The x0, y0, x1, y1 are normalized. We need to map it back.
    """
    x0, y0, x1, y1 = [int(i) for i in [x0, y0, x1, y1]]
    # image = cv2.imread(img_filename, 1)
    # h, w, _ = image.shape
    h, w = Image.open(img_filename).size

    x0, x1 = x0 / 1000 * w, x1 / 1000 * w
    y0, y1 = y0 / 1000 * h, y1 / 1000 * h

    x0, y0, x1, y1 = [str(i) for i in [x0, y0, x1, y1]]
    return x0, y0, x1, y1


def export_to_subdir(root_dir: str, dataset_name: str, file_list: List, src_imgs_dir: str, src_raw_ann_dir: str):
    """
    Export a split sub-dataset (train, val or test) to sub_data_dir (under root_dir).
    It includes:
    1. A sample list `.csv` file;
    2. `boxes_and_transcripts` folder contains annotation file `.tsv`, `images` folder contains `.jpg` file.

    src_imgs_dir: the directory contains images;
    src_raw_ann_dir: the directory contains raw annotations.
    """
    # Prepare the folders.
    utils.mkdir(os.path.join(root_dir, dataset_name))
    img_folder = os.path.join(root_dir, dataset_name, 'images')
    ann_folder = os.path.join(root_dir, dataset_name, 'boxes_and_transcripts')
    utils.mkdir(img_folder)
    utils.mkdir(ann_folder)

    # Copy the images and annotations to the directory.
    with Pool(processes=20) as p:
        for idx, _ in enumerate(p.imap_unordered(
                partial(export_single_example, ann_folder=ann_folder,
                        img_folder=img_folder, src_imgs_dir=src_imgs_dir, src_raw_ann_dir=src_raw_ann_dir),
                file_list
        )):
            # Logging.
            if idx % 1000 == 0:
                print(f'Finish processing {idx} samples in {dataset_name} ...')

    # Write the summary index files.
    # line format: index,document_type,file_name.
    summary_idx_file = os.path.join(root_dir, dataset_name, f'{dataset_name}_samples_list.csv')
    lines = []
    for idx, filename in enumerate(os.listdir(ann_folder), start=1):
        filename = filename.replace(',', '')
        lines.append(f'{idx},document,{filename}\n')

    with open(summary_idx_file, 'w') as file:
        file.writelines(lines)

    print(f'Finish processing dataset {dataset_name} ...')


def export_single_example(ann_filename, ann_folder, img_folder, src_imgs_dir, src_raw_ann_dir):
    ann_basename = os.path.splitext(os.path.basename(ann_filename))[0]
    img_filename = os.path.join(src_imgs_dir, f'{ann_basename}_ori.jpg')
    ann_filename = os.path.join(src_raw_ann_dir, f'{ann_basename}.txt')
    # Check if image or annotation exists.
    assert os.path.isfile(img_filename), f'{img_filename} does not exists!'
    assert os.path.isfile(ann_filename), f'{ann_filename} does not exists!'

    # We process the raw annotation and directly save to new location.
    des_basename = ann_basename.replace(',', '')
    is_success = ann_convert(
        ann_filename,
        os.path.join(ann_folder, f'{des_basename}.tsv'),
        img_filename
    )

    if is_success:
        utils.copy_or_move_file(
            img_filename,
            os.path.join(img_folder, f'{des_basename}.jpg')
        )


def export(des_root_dir: str, idx_files_dir: str, src_imgs_dir: str, src_raw_ann_dir: str):
    """
    Export the images and folders into train, dev and test folders.
    The des_root_dir contains `train/`, `dev/`, `test/`.

    Under the directory `idx_files_dir`, there are: `500K_all.txt`, `500K_dev.txt`, `500K_test.txt`, `500K_train.txt`.

    :param
    :return:
    """
    utils.mkdir(des_root_dir)

    # Export to subdirectories.
    sub_dirs = ['train', 'dev', 'test']
    for sub_dir in sub_dirs:
        with open(os.path.join(idx_files_dir, f'500K_{sub_dir}.txt'), 'r') as file:
            file_list = file.readlines()

        # Remove '\n'.
        file_list = [file_name.strip() for file_name in file_list]
        export_to_subdir(des_root_dir, sub_dir, file_list, src_imgs_dir, src_raw_ann_dir)


def main(args):
    export(
        args.des_dir,
        './indexed_files/',
        args.src_imgs_dir,
        args.src_raw_ann_dir
    )
    # root_dir = '/Users/bytedance/Downloads'
    # des_img_filename = '/Users/bytedance/Downloads/test.jpg'
    # ann_filename = '/Users/bytedance/Downloads/test.tsv'
    # src_img_filename = os.path.join(root_dir, '149.tar_1805.11250.gz_QDAC_QADC_5_ori.jpg')
    # ann_convert(os.path.join(root_dir, '149.tar_1805.11250.gz_QDAC_QADC_5.txt'),
    #             ann_filename, src_img_filename)
    #
    # utils.draw_bboxes_with_ann(src_img_filename, ann_filename, des_img_filename,
    #                            label_to_color={
    #                                'paragraph': (0, 0, 255),
    #                                'caption': (255, 0, 0),
    #                                'option': (0, 255, 0),
    #                                'section': (255, 255, 0),
    #                                'equation': (0, 255, 255),
    #                                'footer': (255, 0, 255),
    #                                'reference': (255, 255, 255),
    #                                'figure': (125, 125, 125)
    #                            })


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
