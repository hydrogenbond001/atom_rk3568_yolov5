# 简介
* 此仓库为c++实现, 大体改自[rknpu2](https://github.com/rockchip-linux/rknpu2),本模型不采用python。
* 使用官方rknn模型, 单线程推理帧数大约20fps。
* 使用使用正点原子的CB1和香橙派5板子测试。

# 更新说明
* 将RK3568与RK3588都可以使用.


# 使用说明
### 演示
  * 系统需安装有**OpenCV** **CMake** **GCC，G++**
  * 下载Releases中的测试视频于项目根目录,运行编译
  ```
  bash build-linux_RK3568.sh
  ```
  
  * 编译完成后进入install运行命令./rknn_yolov5_demo **模型所在路径** **视频所在路径/摄像头序号**
  ```
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn ./bus.jpg
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn 10
  ./rknn_yolov5_demo ./model/yolov5s3568_80.rknn ./*.mp4
  ```

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型来源: 自己转化
* [yolov5s-relu](https://github.com/rockchip-linux/rknpu2/blob/master/examples/rknn_yolov5_demo/model/RK3566_RK3568/yolov5s-640-640.rknn)
* 测试视频可见于 [bilibili](https://www.bilibili.com/video/BV1YvrUYBEZ5/?spm_id_from=333.1007)


# 补充
* 异常处理尚未完善, 目前仅支持rk3588/rk3588s下的运行

# Acknowledgements
* https://github.com/leafqycc/rknn-cpp-Multithreading
* https://github.com/hydrogenbond001/rknn-yolov5-cpp
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo