#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

static const int INFERENCE_SIZE = 224;
static const char* TORCH_MODULE_PATH = "/home/rashad/git/pam-torch-auth/tools/efficientnet_b7.pt";

cv::Mat crop_to_square(const cv::Mat& frame);
at::Tensor preprocess_mat(cv::Mat frame);
at::Tensor to_torch_channels(const at::Tensor& tensor);
at::Tensor normalize_img(at::Tensor tensor, const at::Tensor& mean, const at::Tensor& std);

bool load_module(torch::jit::Module& module);
std::vector<torch::IValue> make_single_input(const at::Tensor& tensor);