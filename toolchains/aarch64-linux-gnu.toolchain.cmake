# 设置目标系统名称
set(CMAKE_SYSTEM_NAME Linux)
# 设置目标系统处理器架构
set(CMAKE_SYSTEM_PROCESSOR aarch64)


# 设置 C 和 C++ 编译器
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

# # 设置 OpenCV 的根目录
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/toolchains/aarch64/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)


# # 打印 OpenCV 版本和路径，方便调试
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libs: ${OpenCV_LIBS}")

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
