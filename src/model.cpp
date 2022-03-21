#include "model.hpp"

cv::Mat crop_to_square(const cv::Mat& frame) {
    const int crop_size = std::min(frame.rows, frame.cols);
    const int offset_w = (frame.cols - crop_size) / 2;
    const int offset_h = (frame.rows - crop_size) / 2;

    const auto center_crop_rect = cv::Rect {offset_w, offset_h, crop_size, crop_size};
    return frame(center_crop_rect).clone();
}

at::Tensor preprocess_mat(cv::Mat frame) {
    frame = crop_to_square(frame);

    cv::Mat frame_scaled;

    cv::resize(frame, frame_scaled, cv::Size(INFERENCE_SIZE, INFERENCE_SIZE), cv::INTER_LINEAR);

    cv::Mat frame_rgb;

    cv::cvtColor(frame_scaled, frame_rgb, cv::COLOR_BGR2RGB);
    frame_rgb.convertTo(frame_rgb, CV_32F);

    auto tensor = at::empty({frame_rgb.rows, frame_rgb.cols, 3});
    std::memcpy(tensor.data_ptr(), frame_rgb.data, sizeof(float) * tensor.numel());

    return tensor;
}

at::Tensor to_torch_channels(const at::Tensor& tensor) {
    return tensor.permute({2, 0, 1});
}

at::Tensor normalize_img(at::Tensor tensor, const at::Tensor& mean, const at::Tensor& std) {
    tensor = tensor.clone();

    tensor /= 255.0f;
    tensor[0] = (tensor[0] - mean[0]) / std[0];
    tensor[1] = (tensor[1] - mean[1]) / std[1];
    tensor[2] = (tensor[2] - mean[2]) / std[2];

    return tensor;
}

bool load_module(torch::jit::Module& module) {
    try {
        module = torch::jit::load(TORCH_MODULE_PATH);
        return true;
    } catch (const c10::Error& e) {
        return false;
    }
}

std::vector<torch::IValue> make_single_input(const at::Tensor& tensor) {
    std::vector<torch::IValue> inputs = {
            tensor.unsqueeze(0)
    };

    return inputs;
}
