#!/bin/bash
# run_test.sh
echo "Running rknn_yolov5_demo..."
chmod +x ./rknn_yolov5_demo run_test.sh
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
# ./rknn_yolov5_demo model/yolov5s3568_80.rknn model/bus.jpg
./rknn_yolov5_demo model/best.rknn 9
