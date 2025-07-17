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

// Shared data structure between threads
struct SharedData
{
    cv::Mat frame;
    std::mutex mutex;
    std::condition_variable cond_var;
    bool frame_ready = false;
    bool processing_done = true;
    std::atomic<bool> exit_flag{false};
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

int main(int argc, char **argv)
{
    if (argc <= 2)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <video_source>" << std::endl;
        return -1;
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
    SharedData shared;

    // Start threads
    std::thread capture_thread(capture_thread_func, std::ref(cap), std::ref(shared));
    std::thread inference_thread(inference_thread_func, ctx, std::ref(shared),
                                 std::cref(io_num), input_attrs.data(), output_attrs.data(),
                                 width, height, channel);

    // Wait for threads to finish
    capture_thread.join();
    inference_thread.join();

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

        // Display results
        // cv::Mat bgr_frame;
        // cv::cvtColor(rgb_frame, bgr_frame, cv::COLOR_RGB2BGR);
        // cv::imshow("Detection", bgr_frame);

        // Release outputs
        rknn_outputs_release(ctx, io_num.n_output, outputs.data());

        // Check for exit key
        if (cv::waitKey(1) >= 0)
        {
            shared.exit_flag = true;
            shared.cond_var.notify_one();
        }

        // Notify capture thread that processing is done
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.processing_done = true;
        }
        shared.cond_var.notify_one();
    }
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