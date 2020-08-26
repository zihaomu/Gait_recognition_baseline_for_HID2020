#!/bin/bash

#CUDA_VISIBLE_DEVICES=4,5 python3 test.py \
#  --config=configs/baseline_config.yml \
#  --epoch=99

CUDA_VISIBLE_DEVICES=6,7 python3 test.py \
  --config=configs/baseline_config_centerloss.yml \
  --epoch=99
