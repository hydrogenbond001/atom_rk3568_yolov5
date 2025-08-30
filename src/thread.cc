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

// Shared data structure between threads
struct SharedData
{
    cv::Mat frame;
    std::mutex mutex;
    std::condition_variable cond_var;
    bool frame_ready = false;
    bool processing_done = true;
    std::atomic<bool> exit_flag{false};

    // 新增串口数据相关成员
    std::mutex serial_mutex;
    std::vector<detect_result_t> detection_results;
    bool new_results_available = false;

    std::atomic<char> flag{0}; // 默认值设为'A'

    // 新增显示相关成员
    cv::Mat display_frame;
    std::mutex display_mutex;
    bool new_frame_for_display = false;
};

struct timeval start_time, stop_time;

// Function prototypes
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
static unsigned char *load_model(const char *filename, int *model_size);
static void dump_tensor_attr(rknn_tensor_attr *attr);
void capture_thread_func(cv::VideoCapture &cap, SharedData &shared);
void inference_thread_func(rknn_context ctx, SharedData &shared,
                           const rknn_input_output_num &io_num,
                           const rknn_tensor_attr *input_attrs,
                           const rknn_tensor_attr *output_attrs,
                           int width, int height, int channel);
void serial_thread_func(SharedData &shared, const std::string &serial_port);
void display_thread_func(SharedData &shared);
int setup_serial_port(const std::string &serial_port, int baud_rate);
bool check_serial_data_available(int fd);

int flag = 0;
int argc_para_flag = 0;
int main(int argc, char **argv)
{
    SharedData shared;
    if (argc <= 2)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <video_path> [serial_port] [flag] [argc_para_flag]" << std::endl;
        std::cerr << "Note: serial_port is optional. If provided, serial transmission will be enabled." << std::endl;
        return -1;
    }

    // 解析可选参数
    bool use_serial = (argc > 3) && (strcmp(argv[3], "") != 0); // 检查串口参数是否提供且非空
    std::string serial_port;
    if (use_serial)
    {
        serial_port = std::string(argv[3]);
    }

    // 解析其他可选参数(flag和argc_para_flag)
    if (argc > 4)
    {
        shared.flag = argv[4][0]; // 设置初始flag值
    }
    if (argc > 5)
    {
        argc_para_flag = argv[5][0];
    }
    // Initialize RKNN model
    int model_size = 0;
    unsigned char *model_data = load_model(argv[1], &model_size);
    if (!model_data)
    {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    if (ret < 0)
    {
        std::cerr << "rknn_init failed: " << ret << std::endl;
        free(model_data);
        return -1;
    }

    // Query model info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        std::cerr << "rknn_query failed: " << ret << std::endl;
        rknn_destroy(ctx);
        free(model_data);
        return -1;
    }

    // Get input attributes
    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            std::cerr << "rknn_query failed: " << ret << std::endl;
            rknn_destroy(ctx);
            free(model_data);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    // Get output attributes
    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&output_attrs[i]);
    }

    // Determine model dimensions
    int width, height, channel;
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

    // Open video source
    cv::VideoCapture cap;
    try
    {
        int camera_id = std::stoi(argv[2]);
        cap.open(camera_id);
    }
    catch (...)
    {
        cap.open(argv[2]);
    }

    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video source" << std::endl;
        rknn_destroy(ctx);
        free(model_data);
        return -1;
    }

    // Shared data between threads
    // SharedData shared;

    // Start threads
    std::thread capture_thread(capture_thread_func, std::ref(cap), std::ref(shared));
    std::thread inference_thread(inference_thread_func, ctx, std::ref(shared),
                                 std::cref(io_num), input_attrs.data(), output_attrs.data(),
                                 width, height, channel);
    // std::thread serial_thread(serial_thread_func, std::ref(shared), std::string(argv[3]));
    std::thread serial_thread;
    if (use_serial)
    {
        serial_thread = std::thread(serial_thread_func, std::ref(shared), serial_port);
    }
    std::thread display_thread(display_thread_func, std::ref(shared));

    // Wait for threads to finish
    capture_thread.join();
    inference_thread.join();
    // serial_thread.join();
    if (use_serial && serial_thread.joinable())
    {
        serial_thread.join();
    }
    display_thread.join();

    // Cleanup
    cap.release();
    rknn_destroy(ctx);
    free(model_data);
    deinitPostProcess();

    return 0;
}

// Video capture thread function
void capture_thread_func(cv::VideoCapture &cap, SharedData &shared)
{
    cv::Mat frame;
    while (!shared.exit_flag)
    {
        cap >> frame;
        if (frame.empty())
        {
            shared.exit_flag = true;
            shared.cond_var.notify_one();
            break;
        }

        // Wait until previous frame is processed
        std::unique_lock<std::mutex> lock(shared.mutex);
        shared.cond_var.wait(lock, [&shared]()
                             { return shared.processing_done || shared.exit_flag; });

        if (shared.exit_flag)
            break;

        // Update shared frame
        frame.copyTo(shared.frame);
        shared.frame_ready = true;
        shared.processing_done = false;
        lock.unlock();
        shared.cond_var.notify_one();
    }
}

// Inference thread function
void inference_thread_func(rknn_context ctx, SharedData &shared,
                           const rknn_input_output_num &io_num,
                           const rknn_tensor_attr *input_attrs,
                           const rknn_tensor_attr *output_attrs,
                           int width, int height, int channel)
{
    // Prepare input tensor
    rknn_input input = {0};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.size = width * height * channel;
    input.fmt = RKNN_TENSOR_NHWC;
    input.pass_through = 0;

    // Prepare output tensors
    std::vector<rknn_output> outputs(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    while (!shared.exit_flag)
    {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(shared.mutex);
            shared.cond_var.wait(lock, [&shared]()
                                 { return shared.frame_ready || shared.exit_flag; });

            if (shared.exit_flag)
                break;
            if (!shared.frame_ready)
                continue;

            frame = shared.frame.clone();
            shared.frame_ready = false;
        }

        // Preprocessing
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

        cv::Mat resized_img;
        cv::Size target_size(width, height);
        float scale_w = (float)width / rgb_frame.cols;
        float scale_h = (float)height / rgb_frame.rows;
        float min_scale = std::min(scale_w, scale_h);
        BOX_RECT pads;

        gettimeofday(&start_time, NULL);

        memset(&pads, 0, sizeof(pads));
        letterbox(rgb_frame, resized_img, pads, min_scale, target_size);

        // Inference
        input.buf = resized_img.data;
        rknn_inputs_set(ctx, 1, &input);
        rknn_run(ctx, nullptr);
        rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr);

        gettimeofday(&stop_time, NULL);
        printf("Inference time: %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

        // Post-processing
        detect_result_group_t detect_result_group;
        std::vector<float> out_scales;
        std::vector<int32_t> out_zps;
        for (int i = 0; i < io_num.n_output; ++i)
        {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }

        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf,
                     height, width, BOX_THRESH, NMS_THRESH, pads,
                     min_scale, min_scale, out_zps, out_scales, &detect_result_group);

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

            cv::rectangle(rgb_frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
            cv::putText(rgb_frame, text, cv::Point(x1, y1 + 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        }

        // 更新检测结果到共享数据
        {
            std::lock_guard<std::mutex> lock(shared.serial_mutex);
            shared.detection_results.clear();
            for (int i = 0; i < detect_result_group.count; i++)
            {
                shared.detection_results.push_back(detect_result_group.results[i]);
            }
            shared.new_results_available = true;
        }

        {
            std::lock_guard<std::mutex> lock(shared.display_mutex);
            cv::cvtColor(rgb_frame, shared.display_frame, cv::COLOR_RGB2BGR);
            shared.new_frame_for_display = true;
        }
        // Release outputs
        rknn_outputs_release(ctx, io_num.n_output, outputs.data());

        // Notify capture thread that processing is done
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.processing_done = true;
        }
        shared.cond_var.notify_one();
    }
}

// 检查串口是否有数据可读
bool check_serial_data_available(int fd)
{
    int bytes;
    ioctl(fd, FIONREAD, &bytes);
    return bytes > 0;
}

// 串口发送线程函数
void serial_thread_func(SharedData &shared, const std::string &serial_port)
{
    // 设置串口
    int serial_fd = setup_serial_port(serial_port, B115200);
    if (serial_fd < 0)
    {
        std::cerr << "Failed to open serial port: " << serial_port << std::endl;
        // 不设置exit_flag，仅退出当前线程
        return;
    }
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
    while (!shared.exit_flag)
    {
        // 检查是否有新的检测结果
        bool has_new_results = false;
        std::vector<detect_result_t> current_results;
        char flag = shared.flag.load(); // 获取当前flag值
        {
            std::lock_guard<std::mutex> lock(shared.serial_mutex);
            if (shared.new_results_available)
            {
                has_new_results = true;
                current_results = shared.detection_results;
                shared.new_results_available = false;
            }
        }

        if (has_new_results == 1)
        { // 2. 处理并发送数据
            if (!current_results.empty())
            {
                char current_flag = shared.flag.load();

                for (const auto &result : current_results)
                {
                    int x = (result.box.left + result.box.right) / 2;
                    int y = (result.box.top + result.box.bottom) / 2;
                    // const char *www = update_and_judge(atoi(result.name), x, y);
                    printf("物料 %s: (%d, %d) ", result.name, x, y);
                    if (y <= 6)
                        continue;

                    uint8_t obj_id = 0;
                    if (strcmp(result.name, "1") == 0 && (current_flag == 'r' || current_flag == 'A'))
                        obj_id = 0x01;
                    else if (strcmp(result.name, "2") == 0 && (current_flag == 'r' || current_flag == 'A'))
                        obj_id = 0x02;
                    else if (strcmp(result.name, "3") == 0 && (current_flag == 'g' || current_flag == 'A'))
                        obj_id = 0x03;
                    else if (strcmp(result.name, "4") == 0 && (current_flag == 'g' || current_flag == 'A'))
                        obj_id = 0x04;
                    else if (strcmp(result.name, "5") == 0 && (current_flag == 'b' || current_flag == 'A'))
                        obj_id = 0x05;
                    else if (strcmp(result.name, "6") == 0 && (current_flag == 'b' || current_flag == 'A'))
                        obj_id = 0x06;
                    else
                        continue;

                    // 构造二进制帧
                    SerialFrame frame;
                    frame.header = 0xAA;
                    frame.obj_id = obj_id;
                    frame.x = static_cast<int16_t>(x);
                    frame.y = static_cast<int16_t>(y);
                    frame.checksum = obj_id ^ (x & 0xFF) ^ ((x >> 8) & 0xFF) ^ (y & 0xFF) ^ ((y >> 8) & 0xFF);
                    frame.footer = 0x55;

                    // 发送帧
                    write(serial_fd, &frame, sizeof(SerialFrame));
                    printf("Sent: ID=0x%02X, X=%d, Y=%d\n", obj_id, x, y);
                }
            }
        }
        // 检查串口是否有数据可读
        if (check_serial_data_available(serial_fd))
        {
            char c;
            ssize_t n = read(serial_fd, &c, 1);
            if (n > 0)
            {
                std::cout << "Received from serial: " << c << std::endl;
                // 这里可以添加对接收数据的处理逻辑
                flag = c;
            }
        }

        // 适当休眠以避免占用太多CPU
        usleep(1000); // 10ms
    }

    // 关闭串口
    close(serial_fd);
}

// 设置串口函数
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

// cv显示线程函数
void display_thread_func(SharedData &shared)
{
    cv::namedWindow("Detection", cv::WINDOW_AUTOSIZE);

    while (!shared.exit_flag)
    {
        cv::Mat frame_to_display;
        bool has_new_frame = false;

        {
            std::lock_guard<std::mutex> lock(shared.display_mutex);
            if (shared.new_frame_for_display)
            {
                frame_to_display = shared.display_frame.clone();
                shared.new_frame_for_display = false;
                has_new_frame = true;
            }
        }

        if (has_new_frame) //&& argc_para_flag
        {
            cv::imshow("Detection", frame_to_display);
            // 检查退出按键
            if (cv::waitKey(1) >= 0)
            {
                shared.exit_flag = true;
            }
        }
        else
        {
            // 没有新帧时适当休眠
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    cv::destroyWindow("Detection");
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        std::cerr << "Failed to open model file: " << filename << std::endl;
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    *model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char *data = (unsigned char *)malloc(*model_size);
    if (!data)
    {
        fclose(fp);
        return nullptr;
    }

    if (fread(data, 1, *model_size, fp) != *model_size)
    {
        fclose(fp);
        free(data);
        return nullptr;
    }

    fclose(fp);
    return data;
}

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