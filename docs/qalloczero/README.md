# Compiling the CPP tree search engine


Activate an environment with torch and install `CMake` and `C++17`.
Then go to `src/qalloczero/alg/ts_cpp_engine/`, make a `build` directory and run, from outside the
run directory,
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```
to generate the Makefile and
```bash
cmake --build build
```
to generate the python bind.

The `cmake` command should have produced a `compile_commands.json`.
In order to make VSCode use it, use this template of `.vscode/c_cpp_properties.json` and adapt it to
your OS and needs
```json
{
    "configurations": [
        {
            "name": "Linux",
            "compileCommands": "${workspaceFolder}/src/qalloczero/alg/ts_cpp_engine/build/compile_commands.json",
            "intelliSenseMode": "linux-gcc-x64",
            "cppStandard": "gnu++17"
        }
    ],
    "version": 4
}
```