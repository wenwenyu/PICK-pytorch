#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse, sys, os, time
from typing import List

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


def ann_convert(src_filename: str, des_filename: str):
    """
    Convert raw annotation files to the files with format fitting PICK.

    Line of source file: transcripts, x0, y0, x1, y1, R, G, B, font, label
    Line of destination file: line_index, box_coordinates, transcripts, label.
    """
    assert os.path.isfile(src_filename), f'{src_filename} does not exists!'
    # Read lines from source file.
    with open(src_filename, 'r') as file:
        src_lines = file.readlines()

    # Write format.
    des_lines = ''
    with open(des_filename, 'w') as file:
        for line_idx, line in enumerate(src_lines):
            transcripts, x0, y0, x1, y1, _, _, _, _, label = line.strip().split('\t')
            des_lines += f'{line_idx},{xyxy_to_box_coord(x0, y0, x1, y1)},{transcripts},{label}\n'

        file.write(des_lines)


def batch_ann_convert(src_dir: str, des_dir: str):
    """
    src_dir contains all source annotation files, des_dir contains all target annotation files.
    """
    assert os.path.exists(src_dir), f'{src_dir} does not exists!'
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)

    # Convert.
    for idx, filename in enumerate(os.listdir(src_dir)):
        if not os.path.isfile(filename) or not filename.endswith('.txt'):
            continue

        try:
            ann_convert(
                os.path.join(src_dir, filename),
                os.path.join(des_dir, filename)
            )
        except:
            print(f'Fail to process {src_dir}!')

        if idx % 1000 == 0:
            print(f'Finish processing {idx} files.')


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

    # Write the summary index files.
    # line format: index,document_type,file_name.
    summary_idx_file = os.path.join(root_dir, dataset_name, f'{dataset_name}_samples_list.csv')
    lines = []
    for idx, filename in enumerate(file_list, start=1):
        filename = filename.split('.')[0]
        lines.append(f'{idx},document,{filename}\n')

    with open(summary_idx_file, 'w') as file:
        file.writelines(lines)

    # Copy the images and annotations to the directory.
    for idx, ann_filename in enumerate(file_list):
        ann_basename = os.path.splitext(os.path.basename(ann_filename))[0]
        img_filename = os.path.join(src_imgs_dir, f'{ann_basename}_ori.jpg')
        ann_filename = os.path.join(src_raw_ann_dir, f'{ann_basename}.txt')

        # Check if image or annotation exists.
        assert os.path.isfile(img_filename), f'{img_filename} does not exists!'
        assert os.path.isfile(ann_filename), f'{ann_filename} does not exists!'

        # We process the raw annotation and directly save to new location.
        try:
            ann_convert(
                ann_filename,
                os.path.join(ann_folder, ann_basename + '.tsv')
            )
            utils.copy_or_move_file(img_filename, img_folder)

        except:
            print(f'Fail to process {ann_filename}  ...')

        # Logging.
        if idx % 1000 == 0:
            print(f'Finish processing {idx} samples in {dataset_name} ...')

    print(f'Finish processing dataset {dataset_name} ...')


def export(des_root_dir: str, idx_files_dir: str, src_imgs_dir: str, src_raw_ann_dir: str):
    """
    Export the images and folders into train, val and test folders.
    The des_root_dir contains `train/`, `dev/`, `test/`.

    Under the directory `idx_files_dir`, there are: `500K_all.txt`, `500K_dev.txt`, `500K_test.txt`, `500K_train.txt`.

    :param
    :return:
    """
    utils.mkdir(des_root_dir)

    # Export to subdirectories.
    sub_dirs = ['train', 'val', 'test']
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
    # batch_ann_convert(
    #     '/Users/bytedance/projects/layout/DocBank/DocBank_samples/DocBank_samples',
    #     '/Users/bytedance/Desktop/tmp_test'
    # )


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
