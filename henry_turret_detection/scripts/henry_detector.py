import sys
import numpy as np
import cv2
import pyzed.sl as sl

def main():
    print("Running henry detection ... Press 'Esc' to quit")
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
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

    camera_infos = zed.get_camera_information()

    image_left = sl.Mat()
    image_depth = sl.Mat()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Retrieve image
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_image(image_depth, sl.VIEW.DEPTH)
                cv2.imshow("ZED", image_depth.get_data())
                key = cv2.waitKey(10)
                if key == 27:    # Esc key to stop
                    break

    cv2.destroyAllWindows()
    image_left.free(sl.MEM.CPU)

    zed.close()

if __name__ == "__main__":
    main()
