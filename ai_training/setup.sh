#!/usr/bin/env bash

sudo apt install -y curl unzip

curl -L "https://app.roboflow.com/ds/7F1h3sKf7v?key=Ebf1rVDoQj" > roboflow.zip
unzip roboflow.zip
rm roboflow.zip

cd yolov5
pip3 install -r requirements.txt

