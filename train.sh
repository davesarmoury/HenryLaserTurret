#!/usr/bin/env bash

cd yolov5/
python train.py --img 416 --batch 16 --epochs 50 --data '../data.yaml' --cfg ../yolo.yaml --weights '' --name yolov5s_results  --cache
