#ifndef _RKNN_YOLOV5_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV5_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.25
#define BOX_THRESH 0.6
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

// 状态位掩码
#define VISIBLE_MASK   0x01  // bit0: 可见标志
#define ROTATE_CW      0x02  // 顺时针旋转
#define ROTATE_CCW     0x04  // 逆时针旋转

#pragma pack(push, 1)
typedef struct
{
    uint8_t header;   // 帧头，固定为0xAA
    uint8_t length;   // 数据部分长度(不包含头尾和校验)
    uint8_t obj_id;   // 目标ID (0x01-0x07)
    int16_t x;        // X坐标(小端格式)
    int16_t y;        // Y坐标(小端格式)
    uint8_t state;    // 状态位
    uint8_t checksum; // 校验和(从length到state的异或)
    uint8_t footer;   // 帧尾，固定为0x55
} SerialFrame;
#pragma pack(pop)

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

void deinitPostProcess();
#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_
