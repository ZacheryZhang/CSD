#!/bin/bash
python3 scripts/zero_shot/unsupervised_representation_learning.py \
    --config_path configs/classification/unsupervised_representation_learning.yaml \
    --experiment_name zero-shot-classification \
    --gpu_id 0
