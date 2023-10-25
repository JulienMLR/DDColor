#!/bin/bash

CUDA_VISIBLE_DEVICES=0
INPUT_PATH="/home/mullerju/Documents/Datasets/DMS_camera/test_david/"
OUTPUT_PATH="/home/mullerju/Documents/Datasets/DMS_camera/output_dd_color/"
MODEL_PATH="/home/mullerju/Documents/DDColor/pretrain/ddcolor_modelscope.pth"

/home/mullerju/anaconda3/envs/ddcolor/bin/python3 inference/colorization_pipline.py \
													--input $INPUT_PATH \
													--output $OUTPUT_PATH \
													--model_path $MODEL_PATH