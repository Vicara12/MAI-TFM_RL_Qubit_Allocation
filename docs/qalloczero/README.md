# Compiling the CPP tree search engine


Activate an environment with torch and install `CMake` and `C++17`.
Then go to `src/qalloczero/alg/ts_cpp_engine/`, make a `build` directory and run, from outside the
run directory,
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```
to generate the Makefile and
```bash
cmake --build build
```
to generate the python bind.