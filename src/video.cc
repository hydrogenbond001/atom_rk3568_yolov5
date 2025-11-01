// this is for video detect
// you can switch them in cmakelist.txt

#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <atomic>
#include "RgaUtils.h"
#include "postprocess.h"
#include "preprocess.h"
#include <sys/time.h>
#include <fcntl.h>   // 文件控制定义
#include <termios.h> // POSIX 终端控制定义
#include <unistd.h>  // UNIX 标准函数定义
#include <sys/ioctl.h>
#include <string.h> // for strerror
#include <errno.h>  // for errno
/*
void send_xy(int serial_fd, int id, int x, int y)
{
#pragma pack(push, 1)
  typedef struct
  {
    uint8_t header;   // 帧头 0xAA
    uint8_t obj_id;   // 对象ID (0x01-0x06)
    int16_t x;        // X坐标
    int16_t y;        // Y坐标
    uint8_t checksum; // 校验和 (obj_id ^ x_low ^ x_high ^ y_low ^ y_high)
    uint8_t footer;   // 帧尾 0x55
  } SerialFrame;
#pragma pack(pop)

  // 构造二进制帧
  SerialFrame frame;
  frame.header = 0xAA;
  frame.obj_id = id; // atoi(det_result->name);
  frame.x = static_cast<int16_t>(x);
  frame.y = static_cast<int16_t>(y);
  frame.checksum = id ^ (x & 0xFF) ^ ((x >> 8) & 0xFF) ^ (y & 0xFF) ^ ((y >> 8) & 0xFF);
  frame.footer = 0x55;

  // 发送帧
  write(serial_fd, &frame, sizeof(SerialFrame));
}
// 物料编号到数组索引的映射（2→0, 4→1, 6→2）B G R
#define MAT_ID_TO_INDEX(id) ((id == 2) ? 0 : ((id == 4) ? 1 : 2)) // 加个错误输入处理

// 判断是否接近稳定点（避免微小抖动误判）
#define IS_STABLE(x, stable_x) (fabs((x) - (stable_x)) < 10)

int history[3][4];

// 初始化历史数据
void init_history()
{
  for (int i = 0; i < 3; i++)
  {
    history[i][0] = -1; // x
    history[i][1] = -1; // y
    history[i][2] = -1; // 稳定点x（未初始化）
    history[i][3] = -1;
  }
}

// 更新数据并判断转动方向
const char update_and_judge(int serial_fd, int id, int x, int y)
{
  int idx = MAT_ID_TO_INDEX(id);
  if (idx == -1)
    printf("Unknown ID"); // 非1/3/5号物料

  float last_x = history[idx][0];
  float last_y = history[idx][1];
  // 如果是第一次检测到该物料，初始化数据
  if (last_x == -1)
  {
    history[idx][0] = x;
    history[idx][1] = y;
    printf("No previous data");
  }

  // 如果不动，视为未转动
  static int stable_count[3] = {0};
  if (IS_STABLE(x, last_x) && IS_STABLE(y, last_y))
  {
    stable_count[idx]++;
    if (stable_count[idx] > 5)
    { // 连续5帧不动则更新稳定点
      history[idx][2] = x;
      history[idx][3] = y;
      stable_count[idx] = 0;
      static char stable_str[32]; // 用于返回稳定点信息的缓冲区
      snprintf(stable_str, sizeof(stable_str), "Stable:%d,%d\r\n", x, y);
      send_xy(serial_fd, 0x09, x, y);
      printf(stable_str);
    }
  }
  else
  {
    stable_count[idx] = 0;
  }

  const char *direction = NULL;
  if (x > last_x + 10)
  {
    direction = "Right"; // 发送rotation_dir
    send_xy(serial_fd, 0x08, idx, 1);
  }
  else if (x < last_x - 10)
  {
    direction = "Left "; // 发送rotation_dir
    send_xy(serial_fd, 0x08, idx, 2);
  }

  // 更新最新坐标
  history[idx][0] = x;
  history[idx][1] = y;

  printf(direction ? direction : "No direction"); // 不用返回ID whois_visible
  printf("\r\n");
}
*/
// Function prototypes
static void dump_tensor_attr(rknn_tensor_attr *attr);
double __get_us(struct timeval t);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i)
  {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
  FILE *fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++)
  {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}
int setup_serial_port(const std::string &serial_port, int baud_rate)
{
  int fd = open(serial_port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
  if (fd == -1)
  {
    perror("open_port: Unable to open serial port");
    return -1;
  }

  // 恢复阻塞模式
  fcntl(fd, F_SETFL, 0);

  struct termios options;
  tcgetattr(fd, &options);

  // 设置波特率
  cfsetispeed(&options, baud_rate);
  cfsetospeed(&options, baud_rate);

  // 启用接收器和本地模式
  options.c_cflag |= (CLOCAL | CREAD);

  // 设置8位数据位，无奇偶校验，1位停止位
  options.c_cflag &= ~PARENB;
  options.c_cflag &= ~CSTOPB;
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8;

  // 禁用硬件流控
  options.c_cflag &= ~CRTSCTS;

  // 原始输入模式
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

  // 原始输出模式
  options.c_oflag &= ~OPOST;

  // 设置超时 - 在15秒内返回，即使没有读取到任何数据
  options.c_cc[VMIN] = 0;
  options.c_cc[VTIME] = 0;

  // 应用设置
  if (tcsetattr(fd, TCSANOW, &options) != 0)
  {
    perror("Error setting serial port attributes");
    close(fd);
    return -1;
  }

  return fd;
}
int main(int argc, char **argv)
{
  if (argc <= 4)
  { // 检查参数数量改为4
    std::cerr << "Usage: " << argv[0] << " <model_path> <video_source> <serial_port> <flag init value>" << std::endl;
    return -1;
  }

  // 设置串口
  int serial_fd = setup_serial_port(argv[3], B115200);
  if (serial_fd < 0)
  {
    std::cerr << "Failed to open serial port: " << argv[3] << std::endl;
  }
  int flag = argv[4][0];

  int ret;
  rknn_context ctx;
  int img_width = 0;
  int img_height = 0;
  int img_channel = 0;
  const float nms_threshold = NMS_THRESH;
  const float box_conf_threshold = BOX_THRESH;
  struct timeval start_time, stop_time;

  char *model_name = (char *)argv[1];
  std::string input = argv[2]; // 获取输入参数
  cv::VideoCapture cap;

  // 判断输入参数是否为数字（摄像头编号）还是文件路径
  try
  {
    int camera_id = std::stoi(input); // 尝试将输入解析为摄像头编号
    cap.open(camera_id);              // 打开摄像头
  }
  catch (...)
  {
    cap.open(input); // 如果解析失败，假定输入为视频文件路径
  }

  printf("Loading model...\n");
  int model_data_size = 0;
  unsigned char *model_data = load_model(model_name, &model_data_size);
  ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  if (ret < 0)
  {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }

  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0)
  {
    printf("rknn_query error ret=%d\n", ret);
    return -1;
  }

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
      printf("rknn_query error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++)
  {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
  }

  int channel = 3;
  int width = 0;
  int height = 0;
  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
  {
    channel = input_attrs[0].dims[1];
    height = input_attrs[0].dims[2];
    width = input_attrs[0].dims[3];
  }
  else
  {
    height = input_attrs[0].dims[1];
    width = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
  }

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = width * height * channel;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;

  cv::Mat img;
  while (true)
  {
    cap >> img;
    if (img.empty())
    {
      printf("Empty frame\n");
      break;
    }

    // Convert to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Resize image if needed
    cv::Mat resized_img;
    cv::Size target_size(width, height);
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;
    float min_scale = std::min(scale_w, scale_h);
    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    letterbox(img, resized_img, pads, min_scale, target_size);

    // Set input data
    inputs[0].buf = resized_img.data;
    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    // Run inference
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
      outputs[i].want_float = 0;
    }
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    // printf("Inference time: %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // Post-process
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
      out_scales.push_back(output_attrs[i].scale);
      out_zps.push_back(output_attrs[i].zp);
    }
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                 box_conf_threshold, nms_threshold, pads, min_scale, min_scale, out_zps, out_scales, &detect_result_group);

    // Draw results
    for (int i = 0; i < detect_result_group.count; i++)
    {
      detect_result_t *det_result = &(detect_result_group.results[i]);
      char text[256];
      sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
      int x1 = det_result->box.left;
      int y1 = det_result->box.top;
      int x2 = det_result->box.right;
      int y2 = det_result->box.bottom;
      int x = (x1 + x2) / 2;
      int y = (y1 + y2) / 2;
      cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
      cv::putText(img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

#pragma pack(push, 1)
      typedef struct
      {
        uint8_t header;   // 帧头 0xAA
        uint8_t obj_id;   // 对象ID (0x01-0x06)
        int16_t x;        // X坐标
        int16_t y;        // Y坐标
        uint8_t checksum; // 校验和 (obj_id ^ x_low ^ x_high ^ y_low ^ y_high)
        uint8_t footer;   // 帧尾 0x55
      } SerialFrame;
#pragma pack(pop)
      // update_and_judge(serial_fd, atoi(det_result->name), x, y);
      // printf("物料 %d: (%d, %d)  \r\n", atoi(det_result->name), x, y);
      printf("物料 %d: (%d, %d)  ", i, x, y);
      if (y <= 6)
        continue;

      uint8_t obj_id = 0;
      if (strcmp(det_result->name, "1") == 0 && (flag == 'r' || flag == 'A'))
        obj_id = 0x01;
      else if (strcmp(det_result->name, "2") == 0 && (flag == 'r' || flag == 'A'))
        obj_id = 0x02;
      else if (strcmp(det_result->name, "3") == 0 && (flag == 'g' || flag == 'A'))
        obj_id = 0x03;
      else if (strcmp(det_result->name, "4") == 0 && (flag == 'g' || flag == 'A'))
        obj_id = 0x04;
      else if (strcmp(det_result->name, "5") == 0 && (flag == 'b' || flag == 'A'))
        obj_id = 0x05;
      else if (strcmp(det_result->name, "6") == 0 && (flag == 'b' || flag == 'A'))
        obj_id = 0x06;
      else
        continue;
      // 构造二进制帧
      SerialFrame frame;
      frame.header = 0xAA;
      frame.obj_id = obj_id; // atoi(det_result->name);
      frame.x = static_cast<int16_t>(x);
      frame.y = static_cast<int16_t>(y);
      frame.checksum = obj_id ^ (x & 0xFF) ^ ((x >> 8) & 0xFF) ^ (y & 0xFF) ^ ((y >> 8) & 0xFF);
      frame.footer = 0x55;

      // 发送帧
      write(serial_fd, &frame, sizeof(SerialFrame));
      // printf("Sent: ID=0x%02X, X=%d, Y=%d\n", obj_id, x, y);
    }
    printf("\r\n");
    // show position

    // Show image
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imshow("Detection", img);

    // Release RKNN outputs
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    // Wait for user input to exit
    if (cv::waitKey(1) >= 0)
      break;

    // Free the resized image memory
    resized_img.release();
  }

  // Cleanup
  cap.release();
  rknn_destroy(ctx);
  cv::destroyAllWindows();
  free(model_data);
  deinitPostProcess(); // Release post-process resources

  return 0;
}
