/*
 * train_cpp.cpp — Native C++ Training for AI Detector
 * ====================================================
 *
 * Uses LibTorch + OpenCV. No Python required.
 * Same CNN architecture as train_pytorch.py's PyTorchCNN.
 *
 * Build:
 *   mkdir -p build && cd build
 *   cmake .. -DCMAKE_PREFIX_PATH="<path-to-libtorch>"
 *   make -j$(nproc)
 *
 * Run:
 *   ./train_cpp --data_dir ../data --epochs 10
 *   ./train_cpp --video ../data/real_videos/clip.mp4 --epochs 5
 *   ./train_cpp --help
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef __linux__
#include <sys/statvfs.h> // disk space
#endif

namespace fs = std::filesystem;

// ════════════════════════════════════════════════════════════════════════════
//  Configuration defaults
// ════════════════════════════════════════════════════════════════════════════

static const int DEFAULT_RESOLUTION = 512;
static const int DEFAULT_BATCH_SIZE = 4;
static const int DEFAULT_GRAD_ACCUM = 8;
static const int DEFAULT_EPOCHS = 10;
static const float DEFAULT_LR = 0.001f;
static const int DEFAULT_FPS_SAMPLE = 1;
static const int DEFAULT_EDITS = 2;

static const std::vector<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png",
                                                    ".bmp", ".tiff", ".webp"};
static const std::vector<std::string> VIDEO_EXTS = {".mp4", ".mov",  ".avi",
                                                    ".mkv", ".webm", ".flv"};

// ════════════════════════════════════════════════════════════════════════════
//  CNN Model (same architecture as Python PyTorchCNN)
// ════════════════════════════════════════════════════════════════════════════

struct PyTorchCNNImpl : torch::nn::Module {
  torch::nn::Sequential features{nullptr};
  torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
  torch::nn::Sequential classifier{nullptr};

  PyTorchCNNImpl() {
    features = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    avgpool = torch::nn::AdaptiveAvgPool2d(
        torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
    classifier = torch::nn::Sequential(
        torch::nn::Flatten(), torch::nn::Linear(64, 64), torch::nn::ReLU(),
        torch::nn::Linear(64, 1), torch::nn::Sigmoid());

    register_module("features", features);
    register_module("avgpool", avgpool);
    register_module("classifier", classifier);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = features->forward(x);
    x = avgpool->forward(x);
    x = classifier->forward(x);
    return x;
  }
};
TORCH_MODULE(PyTorchCNN);

// ════════════════════════════════════════════════════════════════════════════
//  Image / Video Utilities
// ════════════════════════════════════════════════════════════════════════════

static bool has_extension(const std::string &path8,
                          const std::vector<std::string> &exts) {
  std::string ext = fs::path(path8).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  for (auto &e : exts) {
    if (ext == e)
      return true;
  }
  return false;
}

// Load image → CHW float tensor, normalised (ImageNet)
static torch::Tensor load_image(const std::string &path, int resolution) {
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "[WARN] Cannot read: " << path << "\n";
    return {};
  }
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::resize(img, img, {resolution, resolution}, 0, 0, cv::INTER_AREA);
  img.convertTo(img, CV_32FC3, 1.0 / 255.0);

  auto tensor =
      torch::from_blob(img.data, {resolution, resolution, 3}, torch::kFloat32)
          .clone();
  tensor = tensor.permute({2, 0, 1}); // HWC → CHW

  // ImageNet normalisation
  auto mean = torch::tensor({0.485f, 0.456f, 0.406f}).view({3, 1, 1});
  auto std_ = torch::tensor({0.229f, 0.224f, 0.225f}).view({3, 1, 1});
  tensor = (tensor - mean) / std_;
  return tensor;
}

// Extract frames from a video at a given FPS sample rate
static std::vector<cv::Mat>
extract_video_frames(const std::string &path, int fps_sample, int resolution) {
  std::vector<cv::Mat> frames;
  cv::VideoCapture cap(path);
  if (!cap.isOpened()) {
    std::cerr << "[WARN] Cannot open video: " << path << "\n";
    return frames;
  }

  double cam_fps = cap.get(cv::CAP_PROP_FPS);
  if (cam_fps <= 0)
    cam_fps = 30.0;
  int interval = std::max(1, static_cast<int>(cam_fps / fps_sample));

  cv::Mat frame;
  int idx = 0;
  while (cap.read(frame)) {
    if (idx % interval == 0) {
      cv::Mat rgb;
      cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
      cv::resize(rgb, rgb, {resolution, resolution}, 0, 0, cv::INTER_AREA);
      frames.push_back(rgb);
    }
    idx++;
  }
  cap.release();
  return frames;
}

// Convert a cv::Mat (RGB, uint8) → CHW normalised tensor
static torch::Tensor mat_to_tensor(const cv::Mat &rgb, int resolution) {
  cv::Mat f;
  rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);
  auto tensor =
      torch::from_blob(f.data, {resolution, resolution, 3}, torch::kFloat32)
          .clone();
  tensor = tensor.permute({2, 0, 1});
  auto mean = torch::tensor({0.485f, 0.456f, 0.406f}).view({3, 1, 1});
  auto std_ = torch::tensor({0.229f, 0.224f, 0.225f}).view({3, 1, 1});
  return (tensor - mean) / std_;
}

// ════════════════════════════════════════════════════════════════════════════
//  AI-Edit Augmentations (OpenCV, very fast)
// ════════════════════════════════════════════════════════════════════════════

static std::mt19937 rng(std::random_device{}());

static cv::Mat apply_blur_sharpen(const cv::Mat &src) {
  cv::Mat blurred, sharpened;
  int ksize = 2 * (rng() % 5 + 1) + 1; // 3,5,7,9,11
  cv::GaussianBlur(src, blurred, {ksize, ksize}, 0);
  float alpha = 1.5f + (rng() % 200) / 100.0f; // 1.5 – 3.5
  cv::addWeighted(blurred, -alpha + 2.0f, src, alpha - 1.0f, 0, sharpened);
  return sharpened;
}

static cv::Mat apply_noise(const cv::Mat &src) {
  cv::Mat noise(src.size(), src.type());
  double sigma = 5.0 + (rng() % 20);
  cv::randn(noise, 0, sigma);
  cv::Mat out;
  cv::add(src, noise, out);
  return out;
}

static cv::Mat apply_rescale(const cv::Mat &src) {
  float factor = 0.2f + (rng() % 30) / 100.0f; // 0.2 – 0.5
  int sw = std::max(1, static_cast<int>(src.cols * factor));
  int sh = std::max(1, static_cast<int>(src.rows * factor));
  cv::Mat small, big;
  cv::resize(src, small, {sw, sh}, 0, 0, cv::INTER_AREA);
  cv::resize(small, big, src.size(), 0, 0, cv::INTER_CUBIC);
  return big;
}

static cv::Mat apply_color_shift(const cv::Mat &src) {
  cv::Mat out = src.clone();
  // Shift each channel by a random amount
  for (int c = 0; c < 3; c++) {
    double shift = (static_cast<int>(rng() % 50) - 25); // -25 to +25
    cv::Mat ch;
    cv::extractChannel(out, ch, c);
    ch.convertTo(ch, -1, 1.0, shift);
    cv::insertChannel(ch, out, c);
  }
  return out;
}

static cv::Mat apply_bilateral(const cv::Mat &src) {
  cv::Mat out;
  int d = 7 + (rng() % 4) * 2; // 7,9,11,13
  double sigColor = 50.0 + (rng() % 70);
  double sigSpace = 50.0 + (rng() % 70);
  cv::bilateralFilter(src, out, d, sigColor, sigSpace);
  return out;
}

using AugFn = cv::Mat (*)(const cv::Mat &);
static AugFn AUG_POOL[] = {
    apply_blur_sharpen, apply_noise,     apply_rescale,
    apply_color_shift,  apply_bilateral,
};
static const int AUG_POOL_SIZE = 5;

static cv::Mat apply_random_edits(const cv::Mat &src) {
  cv::Mat out = src.clone();
  int n_edits = 1 + (rng() % 3);
  // Randomly pick n_edits distinct augmentations
  std::vector<int> indices(AUG_POOL_SIZE);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  for (int i = 0; i < std::min(n_edits, AUG_POOL_SIZE); i++) {
    out = AUG_POOL[indices[i]](out);
  }
  return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  Data Collection
// ════════════════════════════════════════════════════════════════════════════

struct Sample {
  torch::Tensor tensor;
  float label;
};

static void collect_images(const std::string &dir, float label, int resolution,
                           std::vector<Sample> &out) {
  if (!fs::exists(dir))
    return;
  int count = 0;
  for (auto &entry : fs::recursive_directory_iterator(dir)) {
    if (!entry.is_regular_file())
      continue;
    if (!has_extension(entry.path().string(), IMAGE_EXTS))
      continue;
    auto t = load_image(entry.path().string(), resolution);
    if (t.numel() == 0)
      continue;
    out.push_back({t, label});
    count++;
  }
  if (count > 0)
    std::cout << "  " << dir << ": " << count << " images (label=" << label
              << ")\n";
}

static void collect_videos(const std::string &dir, float label, int resolution,
                           int fps_sample, int edits_per_frame,
                           std::vector<Sample> &out) {
  if (!fs::exists(dir))
    return;
  int vid_count = 0, frame_count = 0;
  for (auto &entry : fs::recursive_directory_iterator(dir)) {
    if (!entry.is_regular_file())
      continue;
    if (!has_extension(entry.path().string(), VIDEO_EXTS))
      continue;

    auto frames =
        extract_video_frames(entry.path().string(), fps_sample, resolution);
    vid_count++;
    for (auto &rgb : frames) {
      // Real frame
      out.push_back({mat_to_tensor(rgb, resolution), label});
      frame_count++;

      // AI-edited variants (label = 1.0)
      if (label == 0.0f) {
        for (int e = 0; e < edits_per_frame; e++) {
          cv::Mat edited = apply_random_edits(rgb);
          out.push_back({mat_to_tensor(edited, resolution), 1.0f});
          frame_count++;
        }
      }
    }
  }
  if (vid_count > 0)
    std::cout << "  " << dir << ": " << vid_count << " videos, " << frame_count
              << " frames\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  Resource Safety Guard
// ════════════════════════════════════════════════════════════════════════════

struct ResourceGuard {
  static constexpr double MIN_DISK_MB = 500.0;
  static constexpr double MIN_RAM_MB = 400.0;
  static constexpr double MAX_RAM_PCT = 93.0;
  static constexpr double MAX_GPU_PCT = 95.0;

  static double get_disk_free_mb(const std::string &path) {
#ifdef __linux__
    struct statvfs stat;
    if (statvfs(path.c_str(), &stat) == 0) {
      return (static_cast<double>(stat.f_bavail) * stat.f_frsize) /
             (1024.0 * 1024.0);
    }
#endif
    return 1e9; // assume OK if can't check
  }

  static double get_ram_free_mb() {
#ifdef __linux__
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
      if (line.rfind("MemAvailable:", 0) == 0) {
        long kb = 0;
        sscanf(line.c_str(), "MemAvailable: %ld kB", &kb);
        return static_cast<double>(kb) / 1024.0;
      }
    }
#endif
    return 1e9;
  }

  static double get_ram_percent() {
#ifdef __linux__
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long total = 0, avail = 0;
    while (std::getline(meminfo, line)) {
      if (line.rfind("MemTotal:", 0) == 0)
        sscanf(line.c_str(), "MemTotal: %ld kB", &total);
      else if (line.rfind("MemAvailable:", 0) == 0)
        sscanf(line.c_str(), "MemAvailable: %ld kB", &avail);
    }
    if (total > 0)
      return ((total - avail) / static_cast<double>(total)) * 100.0;
#endif
    return 0;
  }

  static double get_gpu_percent() {
    if (torch::cuda::is_available()) {
      auto alloc = static_cast<double>(
          torch::cuda::memory_stats(0)["allocated_bytes.all.current"]);
      auto total =
          static_cast<double>(at::cuda::getDeviceProperties(0)->totalGlobalMem);
      if (total > 0)
        return (alloc / total) * 100.0;
    }
    return 0;
  }

  // Returns {is_critical, reason}
  static std::pair<bool, std::string> check(const std::string &models_dir) {
    double disk = get_disk_free_mb(models_dir);
    if (disk < MIN_DISK_MB)
      return {true, "Disk: " + std::to_string(static_cast<int>(disk)) +
                        " MB free (min " +
                        std::to_string(static_cast<int>(MIN_DISK_MB)) + " MB)"};

    double ram = get_ram_free_mb();
    if (ram < MIN_RAM_MB)
      return {true, "RAM: " + std::to_string(static_cast<int>(ram)) +
                        " MB free (min " +
                        std::to_string(static_cast<int>(MIN_RAM_MB)) + " MB)"};

    double ram_pct = get_ram_percent();
    if (ram_pct > MAX_RAM_PCT)
      return {true, "RAM: " + std::to_string(static_cast<int>(ram_pct)) +
                        "% used (max " +
                        std::to_string(static_cast<int>(MAX_RAM_PCT)) + "%)"};

    return {false, ""};
  }
};

static void emergency_save(PyTorchCNN &model, const std::string &models_dir,
                           const std::string &reason) {
  std::cerr << "\n  🛑  EMERGENCY STOP: " << reason << "\n";
  std::cerr << "       Saving model before exit...\n";
  fs::create_directories(models_dir);
  std::string path = models_dir + "/ai_detector_model_EMERGENCY.pth";
  try {
    torch::save(model, path);
    std::cerr << "  💾  Emergency model saved: " << path << "\n";
  } catch (const std::exception &e) {
    std::cerr << "  ⚠  Could not save: " << e.what() << "\n";
  }
  std::cerr << "  ❌  Stopped safely to prevent system freeze.\n";
  std::cerr << "      Re-run with --resume " << path << "\n\n";
  std::exit(1);
}

// ════════════════════════════════════════════════════════════════════════════
//  Training Loop
// ════════════════════════════════════════════════════════════════════════════

static void train(const std::string &data_dir, const std::string &video_path,
                  const std::string &models_dir,
                  const std::string &resume_model, int resolution,
                  int batch_size, int grad_accum, int epochs, float lr,
                  int fps_sample, int edits_per_frame) {
  // ── Device ──
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "🎮 Using CUDA GPU\n";
  } else {
    std::cout << "💻 Using CPU\n";
  }

  // ── Collect data ──
  std::cout << "\n── Collecting Data ──\n";
  std::vector<Sample> samples;

  if (!video_path.empty()) {
    // Single video mode
    auto frames = extract_video_frames(video_path, fps_sample, resolution);
    std::cout << "  Video: " << frames.size() << " frames from " << video_path
              << "\n";
    for (auto &rgb : frames) {
      samples.push_back({mat_to_tensor(rgb, resolution), 0.0f});
      for (int e = 0; e < edits_per_frame; e++) {
        cv::Mat edited = apply_random_edits(rgb);
        samples.push_back({mat_to_tensor(edited, resolution), 1.0f});
      }
    }
  } else {
    // Directory mode
    collect_images(data_dir + "/real", 0.0f, resolution, samples);
    collect_images(data_dir + "/personal", 0.0f, resolution, samples);
    collect_images(data_dir + "/ai", 1.0f, resolution, samples);
    collect_images(data_dir + "/generated_ai/images", 1.0f, resolution,
                   samples);
    collect_images(data_dir + "/edited_ai/images", 1.0f, resolution, samples);

    // Also process any videos in real/ and personal/
    collect_videos(data_dir + "/real", 0.0f, resolution, fps_sample,
                   edits_per_frame, samples);
    collect_videos(data_dir + "/personal", 0.0f, resolution, fps_sample,
                   edits_per_frame, samples);
  }

  if (samples.empty()) {
    std::cerr << "[ERROR] No training data found!\n";
    return;
  }

  // ── Shuffle ──
  std::shuffle(samples.begin(), samples.end(), rng);

  // ── Split: 80% train, 20% val ──
  size_t split = static_cast<size_t>(samples.size() * 0.8);
  if (split == 0)
    split = 1;

  std::cout << "\n  Total: " << samples.size() << " samples"
            << " (train: " << split << ", val: " << samples.size() - split
            << ")\n";

  // ── Model ──
  PyTorchCNN model;
  model->to(device);

  if (!resume_model.empty() && fs::exists(resume_model)) {
    std::cout << "  Loading weights: " << resume_model << "\n";
    try {
      torch::load(model, resume_model);
      std::cout << "  ✓ Fine-tuning mode\n";
    } catch (...) {
      std::cout << "  ⚠ Could not load weights, starting fresh\n";
    }
  }

  auto criterion = torch::nn::BCELoss();
  auto optimizer =
      torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(lr));

  // ── Print config ──
  int effective_batch = batch_size * grad_accum;
  std::cout << "\n════════════════════════════════════════\n"
            << "  C++ Training Pipeline\n"
            << "════════════════════════════════════════\n"
            << "  Resolution      : " << resolution << "×" << resolution << "\n"
            << "  Batch           : " << batch_size << " (effective "
            << effective_batch << ")\n"
            << "  Epochs          : " << epochs << "\n"
            << "  Learning rate   : " << lr << "\n"
            << "  Device          : " << device << "\n"
            << "════════════════════════════════════════\n\n";

  // ── Training loop ──
  for (int epoch = 0; epoch < epochs; epoch++) {
    // ── Safety check at epoch start ──
    auto [critical, reason] = ResourceGuard::check(models_dir);
    if (critical)
      emergency_save(model, models_dir, reason);

    auto epoch_start = std::chrono::steady_clock::now();

    // Shuffle training portion
    std::shuffle(samples.begin(), samples.begin() + split, rng);

    model->train();
    double running_loss = 0.0;
    int correct = 0, total = 0, micro = 0;
    optimizer.zero_grad();

    for (size_t i = 0; i < split; i += batch_size) {
      size_t end = std::min(i + batch_size, split);
      int bs = static_cast<int>(end - i);

      // Stack batch
      std::vector<torch::Tensor> tensors, labels;
      tensors.reserve(bs);
      labels.reserve(bs);
      for (size_t j = i; j < end; j++) {
        tensors.push_back(samples[j].tensor);
        labels.push_back(torch::tensor({samples[j].label}));
      }
      auto input = torch::stack(tensors).to(device);
      auto target = torch::stack(labels).to(device);

      // Forward
      auto output = model->forward(input);
      auto loss = criterion(output, target) / static_cast<float>(grad_accum);

      // Backward
      loss.backward();
      micro++;

      if (micro % grad_accum == 0) {
        optimizer.step();
        optimizer.zero_grad();
      }

      // Metrics
      running_loss += loss.item<float>() * grad_accum * bs;
      auto pred = (output.detach() > 0.5f).to(torch::kFloat32);
      total += bs;
      correct += (pred == target).sum().item<int>();

      // Progress
      if (total > 0 && total % (batch_size * 50) == 0) {
        std::cout << "    samples " << total << " | loss "
                  << (running_loss / total) << " | acc "
                  << (static_cast<float>(correct) / total) << "\n";

        // Safety check every 200 samples
        auto [crit, rsn] = ResourceGuard::check(models_dir);
        if (crit)
          emergency_save(model, models_dir, rsn);
      }
    }

    // Flush remaining gradients
    if (micro % grad_accum != 0) {
      optimizer.step();
      optimizer.zero_grad();
    }

    float epoch_loss = (total > 0) ? (running_loss / total) : 0.0f;
    float epoch_acc =
        (total > 0) ? (static_cast<float>(correct) / total) : 0.0f;

    // ── Validation ──
    model->eval();
    double val_loss_sum = 0.0;
    int val_correct = 0, val_total = 0;
    {
      torch::NoGradGuard no_grad;
      for (size_t i = split; i < samples.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, samples.size());
        int bs = static_cast<int>(end - i);

        std::vector<torch::Tensor> tensors, labels;
        for (size_t j = i; j < end; j++) {
          tensors.push_back(samples[j].tensor);
          labels.push_back(torch::tensor({samples[j].label}));
        }
        auto input = torch::stack(tensors).to(device);
        auto target = torch::stack(labels).to(device);
        auto output = model->forward(input);
        auto loss = criterion(output, target);

        val_loss_sum += loss.item<float>() * bs;
        auto pred = (output > 0.5f).to(torch::kFloat32);
        val_total += bs;
        val_correct += (pred == target).sum().item<int>();
      }
    }

    float val_loss = (val_total > 0) ? (val_loss_sum / val_total) : 0.0f;
    float val_acc =
        (val_total > 0) ? (static_cast<float>(val_correct) / val_total) : 0.0f;

    auto elapsed = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - epoch_start)
                       .count();

    std::cout << "  ✓ Epoch " << (epoch + 1) << "/" << epochs << " | loss "
              << epoch_loss << " | acc " << epoch_acc << " | val_loss "
              << val_loss << " | val_acc " << val_acc << " | " << elapsed
              << "s\n";
  }

  // ── Save model ──
  fs::create_directories(models_dir);

  std::string pth_path = models_dir + "/ai_detector_model_pytorch.pth";
  torch::save(model, pth_path);
  std::cout << "\n  💾 Model saved: " << pth_path << "\n";

  // Save TorchScript
  try {
    std::string ts_path = models_dir + "/ai_detector_model_pytorch_script.ts";
    model->eval();
    model->to(torch::kCPU); // trace on CPU for portability
    auto dummy = torch::randn({1, 3, resolution, resolution});
    auto traced = torch::jit::trace(model, dummy);
    traced.save(ts_path);
    std::cout << "  💾 TorchScript saved: " << ts_path << "\n";
  } catch (const std::exception &e) {
    std::cerr << "  ⚠ TorchScript save failed: " << e.what() << "\n";
  }

  std::cout << "\n════════════════════════════════════════\n"
            << "  Training complete!\n"
            << "════════════════════════════════════════\n\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  CLI
// ════════════════════════════════════════════════════════════════════════════

static void print_help(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n\n"
      << "Options:\n"
      << "  --data_dir DIR       Data root directory (default: data/)\n"
      << "  --video PATH         Train from a single video file\n"
      << "  --models_dir DIR     Where to save models (default: models/)\n"
      << "  --resume PATH        Load existing .pth to fine-tune\n"
      << "  --resolution N       Image size (default: " << DEFAULT_RESOLUTION
      << ")\n"
      << "  --batch_size N       Micro-batch size (default: "
      << DEFAULT_BATCH_SIZE << ")\n"
      << "  --grad_accum N       Gradient accumulation steps (default: "
      << DEFAULT_GRAD_ACCUM << ")\n"
      << "  --epochs N           Number of epochs (default: " << DEFAULT_EPOCHS
      << ")\n"
      << "  --lr F               Learning rate (default: " << DEFAULT_LR
      << ")\n"
      << "  --fps_sample N       Video frame sample rate (default: "
      << DEFAULT_FPS_SAMPLE << ")\n"
      << "  --edits N            AI edits per real frame (default: "
      << DEFAULT_EDITS << ")\n"
      << "  --help               Show this help\n\n"
      << "Examples:\n"
      << "  " << prog << " --data_dir ../data --epochs 10\n"
      << "  " << prog << " --video ../data/real_videos/clip.mp4\n"
      << "  " << prog << " --resume ../models/ai_detector_model_pytorch.pth\n";
}

int main(int argc, char *argv[]) {
  // Parse args
  std::string data_dir = "data";
  std::string video_path;
  std::string models_dir = "models";
  std::string resume;
  int resolution = DEFAULT_RESOLUTION;
  int batch_size = DEFAULT_BATCH_SIZE;
  int grad_accum = DEFAULT_GRAD_ACCUM;
  int epochs = DEFAULT_EPOCHS;
  float lr = DEFAULT_LR;
  int fps_sample = DEFAULT_FPS_SAMPLE;
  int edits = DEFAULT_EDITS;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      return 0;
    } else if (arg == "--data_dir" && i + 1 < argc)
      data_dir = argv[++i];
    else if (arg == "--video" && i + 1 < argc)
      video_path = argv[++i];
    else if (arg == "--models_dir" && i + 1 < argc)
      models_dir = argv[++i];
    else if (arg == "--resume" && i + 1 < argc)
      resume = argv[++i];
    else if (arg == "--resolution" && i + 1 < argc)
      resolution = std::atoi(argv[++i]);
    else if (arg == "--batch_size" && i + 1 < argc)
      batch_size = std::atoi(argv[++i]);
    else if (arg == "--grad_accum" && i + 1 < argc)
      grad_accum = std::atoi(argv[++i]);
    else if (arg == "--epochs" && i + 1 < argc)
      epochs = std::atoi(argv[++i]);
    else if (arg == "--lr" && i + 1 < argc)
      lr = std::atof(argv[++i]);
    else if (arg == "--fps_sample" && i + 1 < argc)
      fps_sample = std::atoi(argv[++i]);
    else if (arg == "--edits" && i + 1 < argc)
      edits = std::atoi(argv[++i]);
    else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_help(argv[0]);
      return 1;
    }
  }

  train(data_dir, video_path, models_dir, resume, resolution, batch_size,
        grad_accum, epochs, lr, fps_sample, edits);

  return 0;
}
