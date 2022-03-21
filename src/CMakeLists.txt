add_library(common STATIC model.cpp)
target_link_libraries(common opencv_core opencv_videoio opencv_imgproc)
target_link_libraries(common "${TORCH_LIBRARIES}")
target_compile_options(common PRIVATE -fPIC)

add_executable(infer infer.cpp)
target_link_libraries(infer common)

add_library(pam_torch SHARED pam_torch.cpp)
target_link_libraries(pam_torch common)

set_target_properties(pam_torch PROPERTIES PREFIX "")
target_compile_options(pam_torch PRIVATE -fPIC -shared -lpam)

