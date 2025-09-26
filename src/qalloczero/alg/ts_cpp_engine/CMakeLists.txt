cmake_minimum_required(VERSION 3.18)
project(ts_cpp_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python (we need Interpreter to query paths, and Development for headers/libpython) ----
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ---- Torch (from the same venv) ----
# Configure with:
#   cmake -S . -B build \
#     -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
find_package(Torch REQUIRED)

# For profiling. Install profiling library with sudo apt install libgoogle-perftools-dev
find_package(Threads REQUIRED)
find_library(GPERFTOOLS_PROFILER profiler)


# Where are the torch shared libs (libtorch.so, libtorch_python.so, ...)? Use the venv’s torch/lib.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
  OUTPUT_VARIABLE TORCH_PY_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch Python lib dir: ${TORCH_PY_LIBDIR}")

# ---- Build the Python extension module ----
add_library(ts_cpp_engine MODULE
  src/inference_server.cpp
  src/tree_search.cpp
  src/ts_engine.cpp
)

# Include headers (PyTorch includes also ship PyTorch’s pybind11)
target_include_directories(ts_cpp_engine PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)

# Link Torch core libs + torch_python + libpython explicitly
target_link_directories(ts_cpp_engine PRIVATE "${TORCH_PY_LIBDIR}")
target_link_libraries(ts_cpp_engine PRIVATE
  ${TORCH_LIBRARIES}
  torch_python
  Python3::Python
  ${GPERFTOOLS_PROFILER} Threads::Threads # For profiling
)

# Ensure the loader finds the torch libs at import time (embed the rpath)
set_target_properties(ts_cpp_engine PROPERTIES
  PREFIX ""
  OUTPUT_NAME "ts_cpp_engine"
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "${TORCH_PY_LIBDIR}"
  INSTALL_RPATH "${TORCH_PY_LIBDIR}"
)

# Match the C++ ABI that your PyTorch was built with (most wheels use 1)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
