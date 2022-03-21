#include <sys/param.h>

#include <pwd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <security/pam_modules.h>
#include <security/pam_appl.h>

#include "model.hpp"
#include <iostream>

extern "C" {

PAM_EXTERN int
pam_sm_authenticate(pam_handle_t *pamh, int flags,
                    int argc, const char *argv[]) {
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Capture couldn't be opened!\n";
        return PAM_PERM_DENIED;
    }


    torch::jit::Module module;
    if (!load_module(module)) {
        std::cerr << "Couldn't load torchscript module!\n";
        return PAM_PERM_DENIED;
    }

    module.eval();
    torch::NoGradGuard no_grad;

    while (true) {
        cv::Mat frame;
        capture.read(frame);

        auto frame_tensor = preprocess_mat(frame);
        frame_tensor = to_torch_channels(frame_tensor);

        const auto mean = at::tensor({0.485, 0.456, 0.406});
        const auto std= at::tensor({0.229, 0.224, 0.225});

        frame_tensor = normalize_img(frame_tensor, mean, std);

        auto output = module.forward(make_single_input(frame_tensor));
        auto idx = at::softmax(output.toTensor(), 1).argmax();

        std::cout << idx << "\n";

        if (idx.item().equal(504)) {
            return PAM_SUCCESS;
        }

        sleep(1);
    }

    return PAM_FAIL_DELAY;
}

PAM_EXTERN int
pam_sm_setcred(pam_handle_t *pamh, int flags,
               int argc, const char *argv[])
{

    return (PAM_SUCCESS);
}

PAM_EXTERN int
pam_sm_acct_mgmt(pam_handle_t *pamh, int flags,
                 int argc, const char *argv[])
{

    return (PAM_SUCCESS);
}

PAM_EXTERN int
pam_sm_open_session(pam_handle_t *pamh, int flags,
                    int argc, const char *argv[])
{

    return (PAM_SUCCESS);
}

PAM_EXTERN int
pam_sm_close_session(pam_handle_t *pamh, int flags,
                     int argc, const char *argv[])
{

    return (PAM_SUCCESS);
}

PAM_EXTERN int
pam_sm_chauthtok(pam_handle_t *pamh, int flags,
                 int argc, const char *argv[])
{

    return (PAM_SERVICE_ERR);
}

#ifdef PAM_MODULE_ENTRY
PAM_MODULE_ENTRY("pam_torch");
#endif

};