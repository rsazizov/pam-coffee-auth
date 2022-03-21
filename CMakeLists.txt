cmake_minimum_required(VERSION 3.0)
project(pam_torch_auth)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 14)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(src)