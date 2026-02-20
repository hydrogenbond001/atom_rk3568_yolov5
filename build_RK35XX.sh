#!/bin/bash
set -e  # 遇到错误立即退出
BASE_DIR=$(pwd)  # 记录当前目录
BUILD_DIR=$BASE_DIR/build  # 设定构建目录

# 选择对应的架构参数
declare -A ARCH_MAP=(
    [1]="build_aarch64:aarch64:"
    [2]="build_cross_aarch64:aarch64:-DCMAKE_TOOLCHAIN_FILE=$BASE_DIR/toolchains/aarch64-linux-gnu.toolchain.cmake"
)

# 显示使用说明
show_usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  1              aarch64 本地编译"
    echo "  2              交叉编译 (x86_64 到 aarch64)"
    echo "  3              清理构建目录"
    echo "  无参数         显示菜单并交互选择"
    echo ""
    echo "示例:"
    echo "  $0 1           # 自动执行本地编译"
    echo "  $0 2           # 自动执行交叉编译"
    echo "  $0 3           # 自动清理构建目录"
    echo "  $0             # 显示菜单手动选择"
}

# 处理命令行参数
if [ $# -gt 0 ]; then
    case "$1" in
        "1"|"2"|"3")
            choice="$1"
            ;;
        "-h"|"--help")
            show_usage
            exit 0
            ;;
        *)
            echo "错误: 无效参数 '$1'"
            show_usage
            exit 1
            ;;
    esac
else
    # 无参数时显示菜单
    echo "请选择编译方式："
    echo "1. aarch64 本地编译"
    echo "2. 交叉编译 (x86_64 到 aarch64)"
    echo "3. 清理构建目录"
    read -p "请输入选项 (1/2/3): " choice
fi

# 执行清理操作
if [[ "$choice" == "3" ]]; then
    echo "清理构建目录..."
    rm -rf "$BUILD_DIR"
    echo "清理完成."
    exit 0
fi

# 获取选择的编译参数
if [[ -z "${ARCH_MAP[$choice]}" ]]; then
    echo "无效选项，请重新运行脚本选择正确的选项。"
    show_usage
    exit 1
fi

IFS=':' read -r BUILD_SUBDIR ARCH TARGET_OPTIONS <<< "${ARCH_MAP[$choice]}"

echo "开始 $([[ "$choice" == "1" ]] && echo "本地编译" || echo "交叉编译")..."

mkdir -p "$BUILD_DIR/$BUILD_SUBDIR"
cd "$BUILD_DIR/$BUILD_SUBDIR"

# 配置和编译
echo "配置CMake..."
cmake $TARGET_OPTIONS "$BASE_DIR" -Wno-dev

echo "开始编译..."
make -j$(nproc)

echo "开始安装..."
make install

echo "编译完成，输出位于 $BUILD_DIR/$BUILD_SUBDIR"