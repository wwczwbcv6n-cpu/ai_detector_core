import os
from setuptools import setup, Extension
import pybind11

# Add OpenCV include and library paths for Fedora/Linux
# Common locations for opencv4 headers and shared objects
opencv_include_dir = '/usr/include/opencv4'
opencv_lib_dir = '/usr/lib64' # on 64-bit Fedora

ext_modules = [
    Extension(
        'fast_video_processor',
        ['src/fast_video_processor.cpp'],
        include_dirs=[
            pybind11.get_include(),
            opencv_include_dir
        ],
        library_dirs=[opencv_lib_dir],
        libraries=['opencv_core', 'opencv_imgproc', 'opencv_video'],
        language='c++',
        extra_compile_args=['-O3', '-Wall', '-shared', '-std=c++14', '-fPIC', '-fvisibility=hidden'],
        extra_link_args=['-Wl,-Bsymbolic'],
    ),
]

setup(
    name='fast_video_processor',
    version='1.0',
    ext_modules=ext_modules,
)
