// video_detector_cpp.cpp
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

namespace py = pybind11;

// Global configuration (matches Python script/train_cpp.cpp)
const int IMG_WIDTH = 512;
const int IMG_HEIGHT = 512;

class VideoDetector {
public:
  VideoDetector(const std::string &model_path) {
    try {
      // Load the TorchScript model
      module = torch::jit::load(model_path);
      module.eval();
      std::cout << "LibTorch model loaded successfully from " << model_path
                << std::endl;
    } catch (const c10::Error &e) {
      std::cerr << "Error loading the model: " << e.msg() << std::endl;
      throw std::runtime_error("Failed to load model");
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
  }

  std::vector<float> detect_video(const std::string &video_path,
                                  int frames_to_sample = 1) {
    // IMPORTANT: Release the GIL while we do heavy C++ processing
    // This prevents the Python interpreter from deadlocking.
    py::gil_scoped_release release;

    std::vector<float> ai_probabilities;
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
      std::cerr << "Could not open video file: " << video_path << std::endl;
      return {};
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0)
      fps = 30.0;
    int skip_interval = (frames_to_sample > 0) ? frames_to_sample : 1;

    std::cout << "Processing video: " << video_path << " ("
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ " << fps << " FPS)"
              << std::endl;

    cv::Mat frame;
    long long frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    auto mean = torch::tensor({0.485f, 0.456f, 0.406f})
                    .view({3, 1, 1})
                    .to(model_device);
    auto std_ = torch::tensor({0.229f, 0.224f, 0.225f})
                    .view({3, 1, 1})
                    .to(model_device);

    while (cap.read(frame)) {
      if (frame_count % skip_interval == 0) {
        // Preprocess: Resize, BGR2RGB, Normalize
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, rgb, {IMG_WIDTH, IMG_HEIGHT}, 0, 0, cv::INTER_AREA);
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        // Convert Mat to Tensor
        auto input_tensor =
            torch::from_blob(rgb.data, {IMG_HEIGHT, IMG_WIDTH, 3},
                             torch::kFloat32)
                .clone();
        input_tensor = input_tensor.to(model_device);
        input_tensor = input_tensor.permute({2, 0, 1}); // HWC -> CHW

        // Normalization
        input_tensor = (input_tensor - mean) / std_;
        input_tensor = input_tensor.unsqueeze(0);

        // Inference
        torch::NoGradGuard no_grad;
        torch::Tensor output = module.forward({input_tensor}).toTensor();

        // Assuming model outputs a single sigmoid probability or we need to
        // apply sigmoid If model already has Sigmoid (like PyTorchCNN in
        // train_cpp.cpp), just use it
        float prob = output.item<float>();
        ai_probabilities.push_back(prob);
      }
      frame_count++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Processed " << frame_count << " frames in "
              << duration.count() << " seconds." << std::endl;

    return ai_probabilities;
  }

private:
  torch::jit::Module module;
  torch::Device model_device = torch::kCPU;
};

PYBIND11_MODULE(video_detector_cpp, m) {
  m.doc() =
      "High-performance video AI detection plugin using LibTorch & OpenCV";
  py::class_<VideoDetector>(m, "VideoDetector")
      .def(py::init<const std::string &>(), py::arg("model_path"))
      .def("detect_video", &VideoDetector::detect_video, py::arg("video_path"),
           py::arg("frames_to_sample") = 1,
           "Process a video file and return AI probabilities for sampled "
           "frames.");
}
