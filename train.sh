#!/bin/bash

#CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
#  --config=configs/baseline_config.yml \

CUDA_VISIBLE_DEVICES=6,7 python3 train.py \
  --config=configs/baseline_config_centerloss.yml \
