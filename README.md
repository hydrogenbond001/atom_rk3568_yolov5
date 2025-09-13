# 简介
* 此仓库为c++实现, 可以跑图片和视频文件/摄像头大体改自[rknpu2](https://github.com/rockchip-linux/rknpu2),本代码不采用python。
* 使用官方rknn模型, rk3568单线程推理帧数大约20fps（50ms左右），3588跑满了720p30hz摄像头，理论上50帧（20ms内）。
* 使用过正点原子的CB1和香橙派5板子测试。
* **在板端和x86上都可编译部署**

# 更新说明
* 将RK356X与RK3588都可以使用.


# 使用说明
### 环境依赖
  * sudo apt install g++-aarch64-linux-gnu
  * sudo apt install gcc-aarch64-linux-gnu
### 演示
  * 系统需安装有**OpenCV** **CMake** **GCC，G++**
  * 下载Releases中的测试视频于项目根目录,运行编译，
  ```
  bash ./build-linux_RK3568.sh
  ```
  
  编译主程序选择，修改CMakelist文件
  ```
  add_executable(rknn_yolov5_demo
		src/rk3568_market.cc
		# src/video.cc     #detect video
		# src/pic.cc        #detect picture
    src/postprocess.cc
    src/preprocess.cc
    #src/rkYolov5s.cc
)
```
选择编译环境:在板子上编译选择1，电脑上编译选择2
```
(base) ph@PH:~/app_project_test/atom_rk3568_yolov5$ ./build-linux_RK35XX.sh 
请选择编译方式：
1. aarch64 本地编译
2. 交叉编译 (x86_64 到 aarch64)
3. 清理构建目录
请输入选项 (1/2/3):2
```


  * 编译完成后cd进入```atom_rk3568_yolov5/install/rknn_yolov5_demo_Linux/```
  
  运行命令./rknn_yolov5_demo **模型所在路径** **图片，视频所在路径/摄像头序号**
  ```
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn ./bus.jpg
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn 10
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn ./*.mp4
  ```

  market2分支需要增加/dev/ttyS3 A 选择输出串口和输出数据类型 A（all） R/G/B （不同颜色的物料和色环，针对工训比赛）
  ```
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn ./bus.jpg /dev/ttyS3 A
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn 10 /dev/ttyS3 A
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn ./*.mp4 /dev/ttyS3 A
  ```
# 帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型来源: 自己转化或官方模型
* [yolov5s-relu](https://github.com/rockchip-linux/rknpu2/blob/master/examples/rknn_yolov5_demo/model/RK3566_RK3568/yolov5s-640-640.rknn)
* 测试视频可见于 [bilibili](https://www.bilibili.com/video/BV1YvrUYBEZ5/?spm_id_from=333.1007)


# 补充
* 异常处理尚未完善, 目前仅支持rk3588/rk3588s下的运行
* 模型转换使用[rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
* 如使用自己训练的模型部署时需要修改model下的coco_80_labels_list.txt文件和postprocess.h中的OBJ_CLASS_NUM

# Acknowledgements
* https://github.com/leafqycc/rknn-cpp-Multithreading
* https://github.com/hydrogenbond001/rknn-yolov5-cpp
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo