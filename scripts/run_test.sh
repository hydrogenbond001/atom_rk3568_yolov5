#!/bin/bash
# run_test.sh
echo "Running rknn_yolov5_demo..."
# chmod +x ./rknn_yolov5_demo run_test.sh
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
<<<<<<< HEAD
./rknn_yolov5_demo model/gongxun.rknn WIN_20241023_13_46_00_Pro.mp4 /dev/ttyS7
=======
<<<<<<< HEAD
./rknn_yolov5_demo model/yolov5s3568_80.rknn 9
>>>>>>> 6fad4282664aae06ae2bb084930a68adaa14e820
# ./rknn_yolov5_demo model/best.rknn 9
=======
# ./rknn_yolov5_demo model/yolov5s3568_80.rknn model/bus.jpg
./rknn_yolov5_demo model/yolov5s3568_80.rknn 9
>>>>>>> 4ceb464b7eac03f4d360bb26d1c155dc9b0e0462
