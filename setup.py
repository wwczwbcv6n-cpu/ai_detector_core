import os
import sys
import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Check if CUDA is available for LibTorch to enable GPU support
LIBTORCH_CUDA_AVAILABLE = False
print("CUDA headers not found, forcing CPU-only build for LibTorch.")


class build_ext(_build_ext):
    """
    Custom build_ext command to handle compiler flags for LibTorch and FFmpeg.
    """
    def build_extensions(self):
        # Determine LibTorch paths
        # This assumes LibTorch is either installed system-wide or pointed to by LIBTORCH_DIR
        libtorch_dir = os.environ.get('LIBTORCH_DIR')
        if libtorch_dir: # If LIBTORCH_DIR is set and not empty
            print(f"Using LIBTORCH_DIR from environment variable: {libtorch_dir}")
        else: # LIBTORCH_DIR is not set or is empty
            try:
                # Try to find LibTorch installation via Python's torch package location
                import torch
                libtorch_dir = os.path.dirname(os.path.dirname(torch.__file__))
                print(f"Discovered LIBTORCH_DIR from torch package: {libtorch_dir}")
            except ImportError:
                print("Warning: LIBTORCH_DIR environment variable not set and torch package not found. "
                      "Please set LIBTORCH_DIR to your LibTorch installation path.")
                sys.exit(1)

        # Check if libtorch_dir exists
        if not os.path.exists(libtorch_dir):
            print(f"Error: LIBTORCH_DIR '{libtorch_dir}' does not exist.")
            sys.exit(1)

        libtorch_include_dir = os.path.join(libtorch_dir, 'include')
        libtorch_lib_dir = os.path.join(libtorch_dir, 'lib')

        if not os.path.exists(libtorch_include_dir) or not os.path.exists(libtorch_lib_dir):
            print(f"Error: LibTorch include or lib directory not found within '{libtorch_dir}'. "
                  f"Expected '{libtorch_include_dir}' and '{libtorch_lib_dir}'.")
            sys.exit(1)
        
        # Determine FFmpeg paths
        # These are common locations, may need adjustment for specific systems
        ffmpeg_include_dirs = [
            '/usr/local/include',
            '/usr/include/ffmpeg',
            '/usr/local/include/ffmpeg',
            '/opt/homebrew/include', # For macOS homebrew
            # Add other common FFmpeg include paths if necessary
        ]
        ffmpeg_library_dirs = [
            '/usr/local/lib',
            '/usr/lib',
            '/opt/homebrew/lib', # For macOS homebrew
            # Add other common FFmpeg library paths if necessary
        ]
        ffmpeg_libraries = [
            'avformat', 'avcodec', 'swscale', 'avutil'
        ]

        # Add LibTorch include and library directories
        for ext in self.extensions:
            ext.include_dirs.append(libtorch_include_dir)
            ext.include_dirs.append(os.path.join(libtorch_include_dir, 'torch', 'csrc', 'api', 'include')) # Specific LibTorch include
            ext.library_dirs.append(libtorch_lib_dir)
            ext.libraries.extend(['torch', 'torch_cpu', 'c10']) # Core LibTorch libraries
            
            if LIBTORCH_CUDA_AVAILABLE:
                ext.libraries.extend(['torch_cuda', 'c10_cuda']) # CUDA-specific libraries
                ext.extra_compile_args.append('-DVIDEO_DETECTOR_USE_CUDA') # Define macro for conditional compilation

            # Add FFmpeg include and library directories
            ext.include_dirs.extend(ffmpeg_include_dirs)
            ext.library_dirs.extend(ffmpeg_library_dirs)
            ext.libraries.extend(ffmpeg_libraries)

            # Add C++17 standard flag
            ext.extra_compile_args.append('-std=c++17')
            ext.extra_compile_args.append('-fPIC') # Position-independent code

            # For debugging (uncomment if needed)
            # ext.extra_compile_args.append('-g') 
            # ext.extra_link_args.append('-g')

        _build_ext.build_extensions(self)

# Define the C++ extension module
video_detector_module = Extension(
    'video_detector_cpp', # Name of the Python module
    sources=['video_detector_cpp.cpp'], # Source file
    include_dirs=[
        # pybind11 includes
        # This will be automatically added by pybind11's get_include()
    ],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[],
    language='c++'
)

setup(
    name='ai_detector_core',
    version='0.1.0',
    author='Jamoliddin Bekyodgorov',
    author_email='jamoliddin@example.com',
    description='A high-performance AI detector core with video processing capabilities.',
    long_description='',
    ext_modules=[video_detector_module],
    cmdclass={'build_ext': build_ext},
    zip_safe=False, # Important for pybind11 modules
    install_requires=[
        'pybind11>=2.6.0',
        'torch>=1.8.0', # Specify minimum torch version if needed
        # Add other dependencies if necessary (e.g., opencv-python for Python-side video processing if any)
    ],
)