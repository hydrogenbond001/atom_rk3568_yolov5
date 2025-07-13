// this is for market robot video detect
// you can switch them in cmakelist.txt

#include <opencv2/opencv.hpp>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fstream>
#include "RgaUtils.h"
#include "postprocess.h"
#include "rknn_api.h"
#include "preprocess.h"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

// #include <errno.h>
// #include <wiringPi.h>
// #include <wiringSerial.h>
// #include <pthread.h>
// #include <stdlib.h>

int start_enable_flag = 0;
int serial_fd = -1;
char serial_fd_received;
// int bytes_read = read(serial_fd, &serial_fd_received, 1);
int uart_init()
{

  serial_fd = open("/dev/ttyS7", O_RDWR | O_NOCTTY | O_NDELAY);
  if (serial_fd == -1)
  {
    perror("无法打开串口");
    return -1;
  }

  struct termios options;

  // 获取当前串口配置
  tcgetattr(serial_fd, &options);

  // 设置波特率 115200
  cfsetispeed(&options, B115200);
  cfsetospeed(&options, B115200);

  // 8N1 配置（8位数据位，无奇偶校验，1个停止位）
  options.c_cflag &= ~PARENB; // 无奇偶校验
  options.c_cflag &= ~CSTOPB; // 1位停止位
  options.c_cflag &= ~CSIZE;
  options.c_cflag |= CS8; // 8位数据位

  // 使配置生效
  tcsetattr(serial_fd, TCSANOW, &options);
}

int x, y;
int serialPuts(int fd, const char *buffer)
{
  int len = strlen(buffer);
  int ret = write(fd, buffer, len);
  if (ret < 0)
  {
    perror("串口发送失败");
    return -1;
  }
  return ret;
}
// 发送整数的函数
void serialPutNumber(const int fd, int num)
{
  char buffer[32];                             // 定义足够大的缓冲区来存储数字字符串
  snprintf(buffer, sizeof(buffer), "%d", num); // 将整数转换为字符串
  serialPuts(fd, buffer);                      // 通过 serialPuts 发送字符串
}

// 发送浮点数的函数
void serialPutFloat(const int fd, float num)
{
  char buffer[32];                               // 定义缓冲区来存储浮点数字符串
  snprintf(buffer, sizeof(buffer), "%.2f", num); // 将浮点数转换为字符串，保留两位小数
  serialPuts(fd, buffer);                        // 通过 serialPuts 发送字符串
}

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

int main(int argc, char **argv)
{
  uart_init();

  int ret;
  rknn_context ctx;
  int img_width = 0;
  int img_height = 0;
  int img_channel = 0;
  const float nms_threshold = NMS_THRESH;
  const float box_conf_threshold = BOX_THRESH;
  struct timeval start_time, stop_time;

  if (argc <= 2)
  {
    std::cerr << "Usage: " << argv[0] << " < modle_address> <media direction>" << std::endl;
    return -1;
  }

  char *model_name = (char *)argv[1];
  // char *video_name = (char *)argv[2];//std::stoi(argv[1]);
  std::string input = argv[2]; // 获取输入参数
  cv::VideoCapture cap;
  cv::VideoWriter writer;

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

  writer.open("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1280, 720));
  if (!writer.isOpened())
  {
    std::cerr << "无法打开输出视频文件写入" << std::endl;
    return -1;
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
    printf("Inference time: %f ms\n\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

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
      x = (x1 + x2) / 2; // 中心点坐标
      y = (y1 + y2) / 2;
      cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
      cv::putText(img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
      cv::line(img, cv::Point(600, 0), cv::Point(600, height), cv::Scalar(0, 0, 255), 2); // 画第二条竖线 (x=600，蓝色，线宽 2)
      cv::line(img, cv::Point(900, 0), cv::Point(900, height), cv::Scalar(255, 0, 0), 2);
      // printf("Detected %s: %.1f%%\n", det_result->name, det_result->prop * 100);
      // printf("x1: %d, y1: %d, x2: %d, y2: %d\n", x1, y1, x2, y2);
      // printf("x: %d, y: %d,\n", x, y);
      // 发送数据到串口

      if ((x > 600) && (x < 800)) // 当物体中心在两条线之中
      {
        if (start_enable_flag == 1) // 收到K开启命令时才使能输出
        {
          // write(serial_fd, &(det_result->name), 1);
          // write(serial_fd, "\n", 1);
          char send_char = det_result->name[0]; // 默认发送第一个字符   5月24日，比赛前改
          // 如果检测到 C、D、E 或 F，则统一发送 'C'
          // if (send_char == 'C' || send_char == 'D' || send_char == 'E' || send_char == 'F' || send_char == 'G' || send_char == 'H' || send_char == 'I' || send_char == 'J')
          // {
          //   send_char = 'C';
            // printf("Detected has been change to %s: %.1f%%    ", send_char, det_result->prop * 100);
          // }
          write(serial_fd, &send_char, 1);
        }
        printf("Detected has been change to %s: %.1f%%    ", det_result->name, det_result->prop * 100);
        // printf("x1: %d, y1: %d, x2: %d, y2: %d\n", x1, y1, x2, y2);
        printf("x: %d, y: %d,\n", x, y);
      }
      // 发送检测到的物体名称和置信度
      // char buffer[128];

      // 组合字符串并发送
      // snprintf(buffer, sizeof(buffer), "Detected %s: %.1f%%\n", det_result->name, det_result->prop * 100);
      // serialPuts(serial_fd, buffer);

      // snprintf(buffer, sizeof(buffer), "x1: %d, y1: %d, x2: %d, y2: %d\n", x1, y1, x2, y2);
      // serialPuts(serial_fd, buffer);
    }

    int bytes_read = read(serial_fd, &serial_fd_received, 1); // 开关
    // bytes_read = 1;                  ////////////////////////////////////////////////////////////////////////
    if (bytes_read > 0)
    {
      // serial_fd_received = 'K';      //////////////////////////////////////////////////////////
      if (serial_fd_received == 'G') // 收到关时start_enable_flag置0
      {
        start_enable_flag = 0;
        printf("start_enable_flag = 0 \r\n");
        // write(serial_fd, "start_enable_flag = 0", strlen("start_enable_flag = 0"));//调试用
        serial_fd_received = '0'; // 归位
      }
      if (serial_fd_received == 'K') // 收到开时start_enable_flag置1
      {
        start_enable_flag = 1;
        printf("start_enable_flag = 1 \r\n");
        // write(serial_fd, "start_enable_flag = 1", strlen("start_enable_flag = 1"));
        serial_fd_received = '0'; // 归位
      }
    }
    // Show image
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    // cv::imshow("Detection", img);
    writer.write(img); // 保存当前帧

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
  writer.release();
  rknn_destroy(ctx);
  cv::destroyAllWindows();
  free(model_data);
  deinitPostProcess(); // Release post-process resources
  close(serial_fd);
  return 0;
}
