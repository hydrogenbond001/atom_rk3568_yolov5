#!/bin/bash
# run_test.sh
echo "Running rknn_yolov5_demo..."
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
# ./rknn_yolov5_demo model/gongxun3576.rknn 9
./rknn_yolov5_demo model/gongxun3576.rknn .mp4 /dev/ttyS1 0
