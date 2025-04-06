#add_definitions("-Wall -g")

cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "-pthread")

if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(STATUS "本地编译")
    find_package(OpenCV REQUIRED)
endif()

# skip 3rd-party lib dependencies
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/include/librknnrt.so)

#rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux//${LIB_ARCH}/librga.so)


include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/3rdparty)
include_directories(${RGA_PATH}/include)




#WiringPi
#find_library(WIRINGPI_LIB wiringPi)
#include_directories(/usr/src/wiringOP,/wiringPi)


# rknn_yolov5_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov5_demo
		src/rk3568_market.cc
		# src/video.cc     #detect video
		# src/pic.cc        #detect picture
    src/postprocess.cc
    src/preprocess.cc
    #src/rkYolov5s.cc
)

target_link_libraries(rknn_yolov5_demo
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  ${RGA_LIB}
  #${WIRINGPI_LIB}
)


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov5_demo DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(PROGRAMS "toolchains/aarch64_so/lib/libopencv_core.so.410" DESTINATION lib)
install(PROGRAMS "toolchains/aarch64_so/lib/libopencv_highgui.so.410" DESTINATION lib)
install(PROGRAMS "toolchains/aarch64_so/lib/libopencv_imgproc.so.410" DESTINATION lib)
install(PROGRAMS "toolchains/aarch64_so/lib/libopencv_imgcodecs.so.410" DESTINATION lib)
install(PROGRAMS "toolchains/aarch64_so/lib/libopencv_videoio.so.410" DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
install(FILES  "scripts/run_test.sh" DESTINATION ./)
install(FILES  "market1.mp4" DESTINATION ./)