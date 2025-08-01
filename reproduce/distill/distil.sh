#!/bin/bash

python3 scripts/distill/distill.py \
  --teaconfig configs/pretraining/pretrain.yaml \
  --stuconfig configs/distill/distill.yaml \
  --gpu_id 0
