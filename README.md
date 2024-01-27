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

## Troubleshooting

Q: Encounter an `unknown error` in runtime

A: Run the following command

```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```
