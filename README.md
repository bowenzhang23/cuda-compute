# Cuda Compute

Matrix and Vector computing library with minimal CUDA.

Example usage in `scripts/run_module.py`

## Install

```bash
# release build
source setup.sh RELEASE
# or debug build (with debug print-outs)
source setup.sh DEBUG
```

## Dependency

`GoogleTest`: for unit testing

`nanobind`: for python binding c++

`numpy`: for performance comparison

## Device Query

```bash
python scripts/device_query.py
```

My Gamer Laptop:

```text
Device Number: 0
NVIDIA GeForce RTX 3060 Laptop GPU
Memory Clock Rate (GHz): 7.001
Memory Bus Width (bits): 192
Peak Memory Bandwidth (GB/s): 336.048
SM Count: 30
Max Threads per Thread Block: 1024
Max Threads per SM: 1536
Max Threads Blocks per SM: 30
StreamPriority: from 0 to -5
ComputeCapability: 8.6
```

## Examples

```bash
source run_scripts.sh
```

## Troubleshooting

### Encounter an `unknown error` in runtime

Run the following command

```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

### Glossary

- Latency: time (ms) from operator start to end
- Bandwidth: processed data in unit time (GB/s)
- Throughput: successful operator in unit time (GFlops)
- Memory Order: see e.g. <https://en.cppreference.com/w/cpp/atomic/memory_order>

### Runtime or Driver API?

Runtime

### Permission issue regarding `ncu` or `nsys`

see <https://developer.nvidia.com/ERR_NVGPUCTRPERM>

### Check GPU details

```bash
nvidia-smi  -q -i <id>
# e.g. Check memory frequency
nvidia-smi  -q -d CLOCK | grep -A 3 "Max Clocks" | grep "Memory"
```

### nvcc flags

In top-level `CMakeLists.txt`:

```cmake
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
```

To set architecture

```cmake
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -arch=sm_86")
```

To set fast math and enable MAD

```cmake
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr --use_fast_math -fmad=true")
```

> see <https://docs.nvidia.com/cuda/archive/11.6.0/pdf/CUDA_Math_API.pdf>

If check the number of registers and the size of smem:

```cmake
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -Xptxas -v")
```

If use OpenMP

```cmake
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -Xcompiler -fopenmp")
```

> and link the library, e.g. `-lgomp`

If compile .cu code to `ptx`

```bash
nvcc --ptx -o main.ptx main.cu
```

### Hyper-Q connection setting

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=32
```
