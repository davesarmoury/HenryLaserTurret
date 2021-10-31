#!/usr/bin/env bash

cd yolov5
python3 detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.4 --source ../test/images
