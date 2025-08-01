#! /bin/bash

python3 scripts/zero_shot/anomaly_detection.py \
    --config_path configs/anomaly_detection/zero_shot.yaml \
    --run_name zero-shot-anomaly-detection \
    --pretraining_run_name easy-salad-135/ \
    --opt_steps 310000 \
    --gpu_id 1
