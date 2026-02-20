cmake_minimum_required(VERSION 3.18)
project(ai_detector_training LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find LibTorch ──
# Pass -DCMAKE_PREFIX_PATH="<path-to-libtorch>" when configuring
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── Find OpenCV ──
find_package(OpenCV REQUIRED)

# ── Training Executable ──
add_executable(train_cpp train_cpp.cpp)
target_link_libraries(train_cpp "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
target_include_directories(train_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})

# Fix LibTorch RPATH on Linux
if(NOT APPLE)
    set_target_properties(train_cpp PROPERTIES
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
        BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()
