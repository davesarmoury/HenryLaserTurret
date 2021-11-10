#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
import sys
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl

import torch.backends.cudnn as cudnn
from numpy import random

sys.path.insert(0, '../yolov5')
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, time_synchronized

from threading import Lock, Thread
from time import sleep

lock = Lock()
new_data = False
exit_signal = False

def zed_thread(svo_filepath=None):

    global image_left, image_depth, exit_signal, new_data, camera_info

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if svo_filepath is not None:
        input_type.set_from_svo_file(svo_filepath)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.depth_maximum_distance = 20

    runtime_params = sl.RuntimeParameters(sensing_mode=sl.SENSING_MODE.FILL, enable_depth=True)

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    camera_info = zed.get_camera_information()

    image_left_tmp = sl.Mat()
    image_depth_tmp = sl.Mat()

    print("Initialized Camera")

    while not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            zed.retrieve_image(image_depth_tmp, sl.VIEW.DEPTH)
            image_left = image_left_tmp.get_data()
            image_depth = image_depth_tmp.get_data()
            lock.release()
            new_data = True

        sleep(0.01)


    print("Cleaning Up Camera")
    image_depth_tmp.free(sl.MEM.CPU)
    image_left_tmp.free(sl.MEM.CPU)
    zed.close()

    print("ZED Thread Dead...")

def main():
    global image_left, image_depth, exit_signal, new_data

    capture_thread = Thread(target=zed_thread, kwargs={'svo_filepath': opt.svo})
    capture_thread.start()

    print("intializing Network...")

    weights, imgsz = opt.weights, opt.img_size
    device = select_device(opt.device)

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 modelimgsz
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    print("Running Henry detection ... Press 'Esc' to quit")

    while True:
        if new_data:
            lock.acquire()
            frame = image_left.copy()
            lock.release()
            new_data = False

            net_image = frame.copy()

            #net_image = net_image[:, :, :-1]
            net_image = net_image[:,:,:3]
            net_image = net_image[0:720, 280:280+720]
            net_image = cv2.resize(net_image, (416, 416))
            net_image = net_image.transpose((2, 0, 1))
            net_image = np.ascontiguousarray(net_image)

            t1 = time_synchronized()
            img = torch.from_numpy(net_image).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            s = ""

#                rospy.loginfo(str(s) + " Done. " + str(t2 - t1) + "s")

                # Process detections
      #      print ("Found " + str(len(pred)) + " Henries", end='\r')
            for i, det in enumerate(pred):
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        if conf > 0.85:
                            p_scaled = [xyxy[0] * 720.0/416.0 + 280, xyxy[1] * 720.0/416.0, xyxy[2] * 720.0/416.0 + 280, xyxy[3] * 720.0/416.0]
                            cv2.rectangle(frame, (int(p_scaled[0]),  int(p_scaled[1])), (int(p_scaled[2]), int(p_scaled[3])), (0,0,255), 3)

#cv2.rectangle(frame, start_point, end_point, color, thickness)
                s += '%gx%g ' % img.shape[2:]  # print string

        #        Z = depth(v,u)
        #        X = (u-cx)*depth/fx
        #        Y = (v-cy)*depth/fy
            cv2.imshow("ZED", frame)

            key = cv2.waitKey(5)
            if key == 27:    # Esc key to stop
                exit_signal = True
                break

            sleep(0.01)

    sleep(5)
    cv2.destroyAllWindows()
    sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
