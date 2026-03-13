#!/bin/bash

# Default to current directory for the source file
SOURCE_FILE="video_detector_cpp.cpp"
OUTPUT_FILE="video_detector_cpp$(python3-config --extension-suffix)"

echo "--- Building $SOURCE_FILE ---"

# 1. Detect PyBind11 includes
PYBIND_INC=$(python3 -m pybind11 --includes)
if [ -z "$PYBIND_INC" ]; then
    echo "Error: pybind11 not found. Please run 'pip install pybind11'"
    exit 1
fi

# 2. Detect LibTorch paths
# We use the torch python module to find its include and lib paths
TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TORCH_INC="-I$TORCH_PATH/include -I$TORCH_PATH/include/torch/csrc/api/include"
TORCH_LIBS="-L$TORCH_PATH/lib -ltorch -ltorch_cpu -lc10"

if [ -d "$TORCH_PATH/lib/libcudart.so" ] || [ -f "$TORCH_PATH/lib/libcudart.so" ]; then
    TORCH_LIBS="$TORCH_LIBS -ltorch_cuda"
fi

# 3. Detect OpenCV flags
OPENCV_CFLAGS=$(pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv 2>/dev/null)
OPENCV_LIBS=$(pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv 2>/dev/null)

if [ -z "$OPENCV_LIBS" ]; then
    echo "Error: OpenCV not found via pkg-config."
    exit 1
fi

# 4. Final Compile Command
CMD="g++ -O3 -shared -std=c++17 -fPIC \
    $PYBIND_INC \
    $TORCH_INC \
    $OPENCV_CFLAGS \
    $SOURCE_FILE \
    -o $OUTPUT_FILE \
    $TORCH_LIBS \
    $OPENCV_LIBS \
    -Wl,-rpath,$TORCH_PATH/lib"

echo "Running command:"
echo "$CMD"
eval "$CMD"

if [ $? -eq 0 ]; then
    echo "--- Build Successful! Created $OUTPUT_FILE ---"
else
    echo "--- Build Failed ---"
    exit 1
fi
