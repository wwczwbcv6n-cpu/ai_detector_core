// video_detector_cpp.cpp
#ifdef VIDEO_DETECTOR_USE_CUDA
#include <ATen/cuda/CUDAContext.h> // Required for at::cuda::is_available()
#endif
#include <torch/script.h>
#include <torch/torch.h>
#include <chrono>                  // For timing
#include <vector>

// FFmpeg headers
// These would typically come from an FFmpeg installation
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// Global configuration (matches Python script)
const int IMG_WIDTH = 512;
const int IMG_HEIGHT = 512;

// Function to initialize FFmpeg (conceptual)
void init_ffmpeg() {
  static bool initialized = false;
  if (!initialized) {
    // No explicit av_register_all() needed for modern FFmpeg versions,
    // but network initialization might be useful.
    avformat_network_init();
    initialized = true;
  }
}

// Function to preprocess a raw frame (conceptual, assuming RGB 8-bit image
// data)
torch::Tensor preprocess_frame_data(const uint8_t *frame_data, int width,
                                    int height, int stride) {
  // Convert raw C++ data (HWC, uint8) to a LibTorch tensor (CHW, float32,
  // normalized)
  torch::Tensor tensor =
      torch::from_blob((void *)frame_data, // cast to non-const void*
                       {height, width, 3}, // HWC layout
                       torch::kU8          // Data type of the blob
      );
  tensor = tensor.to(torch::kFloat32).div(255.0); // Normalize to [0, 1]
  tensor = tensor.permute({2, 0, 1});             // HWC -> CHW

  // Apply ImageNet normalization (must match Python training pipeline)
  auto mean = torch::tensor({0.485f, 0.456f, 0.406f}).view({3, 1, 1});
  auto std = torch::tensor({0.229f, 0.224f, 0.225f}).view({3, 1, 1});
  tensor = (tensor - mean) / std;

  // Add batch dimension
  tensor = tensor.unsqueeze(0);
  return tensor;
}

// Main C++ class that combines video processing and model inference
class VideoDetector {
public:
  VideoDetector(const std::string &model_path) {
    try {
      // Load the TorchScript model
      module = torch::jit::load(model_path);
      module.eval(); // Set to evaluation mode
      std::cout << "LibTorch model loaded successfully from " << model_path
                << std::endl;
    } catch (const c10::Error &e) {
      std::cerr << "Error loading the model: " << e.msg() << std::endl;
      exit(EXIT_FAILURE);
    }
    // Move model to GPU if available
    if (torch::cuda::is_available()) {
      model_device = torch::kCUDA;
      module.to(model_device);
      std::cout << "Model moved to CUDA GPU." << std::endl;
    } else {
      model_device = torch::kCPU;
      module.to(model_device);
      std::cout << "CUDA not available, running on CPU." << std::endl;
    }
    init_ffmpeg(); // Initialize FFmpeg once
  }

  // Process a video file and return detection results
  std::vector<float> detect_video(const std::string &video_path,
                                  int frames_to_sample = 1) {
    std::vector<float> ai_probabilities;

    AVFormatContext *format_ctx = nullptr;
    AVCodecContext *codec_ctx = nullptr;
    SwsContext *sws_ctx = nullptr;
    AVFrame *frame = nullptr;
    AVFrame *rgb_frame = nullptr;
    AVPacket *packet = nullptr;

    try {
      // 1. Open video file
      if (avformat_open_input(&format_ctx, video_path.c_str(), nullptr,
                              nullptr) < 0) {
        std::cerr << "Could not open video file: " << video_path << std::endl;
        return {};
      }
      if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream information." << std::endl;
        return {};
      }

      // 2. Find video stream
      int video_stream_idx = -1;
      for (unsigned int i = 0; i < format_ctx->nb_streams; ++i) {
        if (format_ctx->streams[i]->codecpar->codec_type ==
            AVMEDIA_TYPE_VIDEO) {
          video_stream_idx = i;
          break;
        }
      }
      if (video_stream_idx == -1) {
        std::cerr << "Could not find a video stream." << std::endl;
        return {};
      }

      AVCodecParameters *codec_params =
          format_ctx->streams[video_stream_idx]->codecpar;
      const AVCodec *codec = avcodec_find_decoder(codec_params->codec_id);
      if (!codec) {
        std::cerr << "Unsupported codec!" << std::endl;
        return {};
      }

      codec_ctx = avcodec_alloc_context3(codec);
      if (!codec_ctx) {
        std::cerr << "Could not allocate codec context." << std::endl;
        return {};
      }
      if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
        std::cerr << "Could not copy codec parameters to context." << std::endl;
        return {};
      }
      if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec." << std::endl;
        return {};
      }

      // 3. Allocate frames and packet
      frame = av_frame_alloc();
      rgb_frame = av_frame_alloc();
      packet = av_packet_alloc();
      if (!frame || !rgb_frame || !packet) {
        std::cerr << "Could not allocate frame or packet." << std::endl;
        return {};
      }

      // Set up SWS context for scaling and color conversion
      // SWS_BILINEAR for scaling, AV_PIX_FMT_RGB24 as target format
      sws_ctx = sws_getContext(codec_ctx->width, codec_ctx->height,
                               codec_ctx->pix_fmt, IMG_WIDTH, IMG_HEIGHT,
                               AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr,
                               nullptr);
      if (!sws_ctx) {
        std::cerr << "Could not initialize SwsContext." << std::endl;
        return {};
      }

      // Allocate RGB buffer for scaled frame
      int num_bytes =
          av_image_get_buffer_size(AV_PIX_FMT_RGB24, IMG_WIDTH, IMG_HEIGHT, 1);
      std::vector<uint8_t> rgb_buffer(num_bytes);
      av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize,
                           rgb_buffer.data(), AV_PIX_FMT_RGB24, IMG_WIDTH,
                           IMG_HEIGHT, 1);

      long long frame_count = 0;
      // Calculate frames to skip based on desired sample rate
      // frames_to_sample = 1 means process every frame (no skip)
      // frames_to_sample > 1 means process 1 frame out of 'frames_to_sample'
      int skip_interval = (frames_to_sample > 0) ? frames_to_sample : 1;

      std::cout << "Starting video processing: " << video_path
                << " (Original: " << codec_ctx->width << "x"
                << codec_ctx->height << " @ "
                << (double)codec_ctx->framerate.num / codec_ctx->framerate.den
                << " FPS)" << std::endl;
      std::cout << "Target frame size: " << IMG_WIDTH << "x" << IMG_HEIGHT
                << std::endl;
      if (skip_interval == 1) {
        std::cout << "Processing all frames." << std::endl;
      } else {
        std::cout << "Sampling 1 frame every " << skip_interval << " frames."
                  << std::endl;
      }

      auto start_time = std::chrono::high_resolution_clock::now();

      while (av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_idx) {
          int response = avcodec_send_packet(codec_ctx, packet);
          if (response < 0 && response != AVERROR(EAGAIN) &&
              response != AVERROR_EOF) {
            std::cerr << "Error sending packet to decoder: " << response
                      << std::endl;
            break;
          }

          while (response >= 0) {
            response = avcodec_receive_frame(codec_ctx, frame);
            if (response < 0) {
              if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                break;
              }
              std::cerr << "Error receiving frame from decoder: " << response
                        << std::endl;
              goto end_of_loop;
            }

            // Process frame if it's a sampled one
            if (frame_count % skip_interval == 0) {
              // Convert and scale frame to RGB24
              sws_scale(sws_ctx, frame->data, frame->linesize, 0,
                        codec_ctx->height, rgb_frame->data,
                        rgb_frame->linesize);

              // Preprocess and infer
              torch::Tensor input_tensor =
                  preprocess_frame_data(rgb_frame->data[0], IMG_WIDTH,
                                        IMG_HEIGHT, rgb_frame->linesize[0]);
              input_tensor = input_tensor.to(
                  model_device); // Ensure tensor is on same device as model

              torch::Tensor output = module.forward({input_tensor}).toTensor();
              float ai_prob = output.item<float>();
              ai_probabilities.push_back(ai_prob);
              // std::cout << "Frame " << frame_count << ": AI Prob = " <<
              // ai_prob << std::endl;
            }
            frame_count++;
            av_frame_unref(frame); // Release frame
          }
        }
        av_packet_unref(packet); // Release packet
      }

      // Flush the decoder
      avcodec_send_packet(codec_ctx, nullptr);
      while (true) {
        int response = avcodec_receive_frame(codec_ctx, frame);
        if (response == AVERROR_EOF) {
          break;
        }
        if (response < 0 && response != AVERROR(EAGAIN)) {
          std::cerr << "Error flushing decoder: " << response << std::endl;
          break;
        }
        if (response >= 0) {
          if (frame_count % skip_interval ==
              0) { // Process flush frame if sampled
            sws_scale(sws_ctx, frame->data, frame->linesize, 0,
                      codec_ctx->height, rgb_frame->data, rgb_frame->linesize);
            torch::Tensor input_tensor =
                preprocess_frame_data(rgb_frame->data[0], IMG_WIDTH, IMG_HEIGHT,
                                      rgb_frame->linesize[0]);
            input_tensor = input_tensor.to(model_device);
            torch::Tensor output = module.forward({input_tensor}).toTensor();
            float ai_prob = output.item<float>();
            ai_probabilities.push_back(ai_prob);
          }
          frame_count++;
        }
        av_frame_unref(frame);
      }

    end_of_loop:; // Label for goto

      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end_time - start_time;
      std::cout << "Processed " << frame_count << " total frames in "
                << duration.count() << " seconds." << std::endl;
      if (!ai_probabilities.empty()) {
        std::cout << "Inferred " << ai_probabilities.size()
                  << " sampled frames." << std::endl;
      }

    } catch (const c10::Error &e) {
      std::cerr << "LibTorch error during inference: " << e.msg() << std::endl;
      ai_probabilities.clear(); // Indicate failure
    } catch (const std::exception &e) {
      std::cerr << "Standard exception during video processing: " << e.what()
                << std::endl;
      ai_probabilities.clear(); // Indicate failure
    }

    // Cleanup FFmpeg resources
    if (packet)
      av_packet_free(&packet);
    if (frame)
      av_frame_free(&frame);
    if (rgb_frame)
      av_frame_free(&rgb_frame);
    if (codec_ctx)
      avcodec_free_context(&codec_ctx);
    if (format_ctx)
      avformat_close_input(&format_ctx);
    if (sws_ctx)
      sws_freeContext(sws_ctx);

    return ai_probabilities;
  }

private:
  torch::jit::Module module;
  torch::Device model_device = torch::kCPU; // Initialize to CPU by default
};

// Pybind11 wrapper
#include <pybind11/chrono.h> // For std::chrono types if needed
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector conversion

namespace py = pybind11;

PYBIND11_MODULE(video_detector_cpp, m) {
  m.doc() =
      "pybind11 plugin for high-performance video AI detection"; // optional
                                                                 // module
                                                                 // docstring

  py::class_<VideoDetector>(m, "VideoDetector")
      .def(py::init<const std::string &>(), py::arg("model_path"))
      .def("detect_video", &VideoDetector::detect_video, py::arg("video_path"),
           py::arg("frames_to_sample") = 1,
           "Process a video file and return AI probabilities for sampled "
           "frames.");
}
