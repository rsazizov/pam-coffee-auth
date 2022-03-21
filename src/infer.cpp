#include "model.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "std_image_write.h"

static const char* LAST_BMP_PATH = "last.bmp";
static const int CV_CAPTURE_DEVICE = 0;

int main() {
    cv::VideoCapture capture(CV_CAPTURE_DEVICE);
    
    if (!capture.isOpened()) {
        std::cerr << "Capture couldn't be opened!\n";
        return -1;
    }

    cv::Mat frame;
    capture.read(frame);

    auto frame_tensor = preprocess_mat(frame);

    stbi_write_bmp(LAST_BMP_PATH, INFERENCE_SIZE, INFERENCE_SIZE, 3,
                   frame_tensor.to(torch::kInt8).data_ptr());

    frame_tensor = to_torch_channels(frame_tensor);

    const auto mean = at::tensor({0.485, 0.456, 0.406});
    const auto std= at::tensor({0.229, 0.224, 0.225});

    frame_tensor = normalize_img(frame_tensor, mean, std);

    torch::jit::Module module;
    if (!load_module(module)) {
        std::cerr << "Couldn't load torchscript module!\n";
        return -1;
    }

    module.eval();
    torch::NoGradGuard no_grad;

    auto output = module.forward(make_single_input(frame_tensor));
    auto idx = at::softmax(output.toTensor(), 1).argmax();

    std::cout << idx << "\n";

    return 0;
}