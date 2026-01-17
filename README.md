# RL Qubit Allocation

The scaling of quantum processors is currently limited by technical challenges such as decoherence and
cross-talk.
Distributed quantum computing deals with these limitations by interconnecting smaller and easier to
handle quantum cores, but it introduces the challenge of minimizing slow and error-prone inter-core
communications.
This work addresses the Qubit Allocation problem, a discrete optimization task aiming to distribute
quantum circuits across cores to minimize communication costs.

Classical algorithms like the Hungarian Qubit Allocation (HQA) currently represent the state of the
art, but they suffer from poor scalability with circuit size.
In contrast, existing Reinforcement Learning (RL) approaches often lack flexibility, requiring
retraining if hardware configurations change, and are far from the solution quality of classical
methods.

To overcome these limitations, this thesis proposes a flexible, transformer-based architecture
capable of handling arbitrary numbers of qubits and cores without retraining.
The policy is trained using a combination of REINFORCE and Group Relative Policy Optimization (GRPO),
two well known RL algorithms.
Furthermore, to explore the test-time compute paradigm applied to this problem, we
introduce QAllocZero, an AlphaZero-inspired tree search algorithm implemented in C++ that utilizes
the trained policy as a heuristic.

The results show that the trained policy achieves up to 25\% improvement in allocation cost with
respect to the state of the art (HQA) and narrows the gap between RL and HQA for most common
circuits while maintaining scalable execution time.
These findings demonstrate that learning-based approaches can effectively break the computational
bottlenecks of classical algorithms, a crucial step for managing future systems with millions of
qubits.

![allocation](data/resources/allocation.png)

## Compiling the CPP tree search engine

Activate an environment with torch and install `CMake` and `C++17`.
Then go to `src/qalloczero/alg/ts_cpp_engine/`, make a `build` directory and run, from outside the
run directory,
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```
to generate the Makefile and
```bash
cmake --build build -j4
```
to generate the python bind.

The `cmake` command should have produced a `compile_commands.json`.
In order to make VSCode use it, use this template of `.vscode/c_cpp_properties.json` and adapt it to
your OS and needs (remember to also install the C/C++ VSCode extension)
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

To get profiling information call `TSCppEngine` with the profile argument to true and run the
following command
```bash
pprof --pdf src/qalloczero/alg/ts_cpp_engine/build/ts_cpp_engine.so profile.prof > report.pdf
```
You might need to install perftools, go language and pprof 
```bash
sudo apt install libgoogle-perftools-dev
sudo apt  install golang-go
go install github.com/google/pprof@latest
mv ~/go/bin/pprof /usr/bin/
```

NOTE: If your system does not have cuda installed (this might be the case if you
have an integrated graphics card), uninstall cuda
```bash
pip uninstall cuda
```
and install a wheel that does not contain cuda via
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```


## Training the model

To properly parallelize training across multiple workers run this command
before executing the python script
```bash
nvidia-cuda-mps-control -d
```
and when you're done run
```bash
echo quit | nvidia-cuda-mps-control
```

If using a large group size you might also need to execute
```bash
ulimit -n 4096
```