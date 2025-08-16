#!/bin/bash
# run_test.sh
echo "Running rknn_yolov5_demo..."
# chmod +x ./rknn_yolov5_demo run_test.sh
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
./rknn_yolov5_demo model/gongxun.rknn WIN_20241023_13_46_00_Pro.mp4 /dev/ttyS7
# ./rknn_yolov5_demo model/best.rknn 9
