#!/bin/bash
set -e  # 遇到错误立即退出
BASE_DIR=$(pwd)  # 记录当前目录
BUILD_DIR=$BASE_DIR/build  # 设定构建目录

# 显示菜单
echo "请选择编译方式："
echo "1. aarch64 本地编译"
echo "2. 交叉编译 (x86_64 到 aarch64)"
echo "3. 清理构建目录"
read -p "请输入选项 (1/2/3): " choice

# 选择对应的架构参数
# 选择对应的架构参数
declare -A ARCH_MAP=(
    [1]="build_aarch64:aarch64:"
    [2]="build_cross_aarch64:aarch64:-DCMAKE_TOOLCHAIN_FILE=$BASE_DIR/toolchains/aarch64-linux-gnu.toolchain.cmake"
)

if [[ "$choice" == "3" ]]; then
    echo "清理构建目录..."
    rm -rf "$BUILD_DIR"
    echo "清理完成."
    exit 0
fi

# 获取选择的编译参数
if [[ -z "${ARCH_MAP[$choice]}" ]]; then
    echo "无效选项，请重新运行脚本选择正确的选项。"
    exit 1
fi

IFS=':' read -r BUILD_SUBDIR ARCH TARGET_OPTIONS <<< "${ARCH_MAP[$choice]}"

mkdir -p "$BUILD_DIR/$BUILD_SUBDIR"
cd "$BUILD_DIR/$BUILD_SUBDIR"

# 配置和编译
cmake $TARGET_OPTIONS "$BASE_DIR"
make -j4

echo "编译完成，输出位于 $BUILD_DIR/$BUILD_SUBDIR"








# # TARGET_SOC="rk3568/rk3588"
# GCC_COMPILER=aarch64-linux-gnu

# export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
# export CC=${GCC_COMPILER}-gcc
# export CXX=${GCC_COMPILER}-g++

# ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# # build
# BUILD_DIR=${ROOT_PWD}/build

# if [ ! -d "${BUILD_DIR}" ]; then
#   mkdir -p ${BUILD_DIR}
# fi

# cd ${BUILD_DIR}
# cmake .. -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_TOOLCHAIN_FILE=../toolchains/toolchain.cmake
# make -j8
# make install
# cd -

# # relu版本
# #cd install/rknn_yolov5_demo_Linux/ && ./rknn_yolov5_demo ./model/RK3588/yolov5s-640-640.rknn ../../720p60hz.mp4
# # 使用摄像头
# # cd install/rknn_yolov5_demo_Linux/ && ./rknn_yolov5_demo ./model/RK3588/yolov5s-640-640.rknn 0

