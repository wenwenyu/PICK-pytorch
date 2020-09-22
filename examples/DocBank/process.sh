#!/bin/bash

# Env variables
BASE_DATA_DIR=/data/layout/DocBank/
SRC_ANN_DIR=${BASE_DATA_DIR}DocBank_500K_txt/
DES_DATA_DIR=${BASE_DATA_DIR}annotations/
SRC_IMG_DIR=${BASE_DATA_DIR}DocBank_500K_ori_img/

# Clean up.
rm -rf ${BASE_DATA_DIR}/train/*
rm -rf ${BASE_DATA_DIR}/dev/*
rm -rf ${BASE_DATA_DIR}/test/*

# Process
python3 DocBank.py --src_dir ${SRC_ANN_DIR}  --des_dir ${BASE_DATA_DIR} \
--src_imgs_dir ${SRC_IMG_DIR} --src_raw_ann_dir ${SRC_ANN_DIR}