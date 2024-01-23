# Cuda Compute

Matrix and Vector computing library with CUDA.

## Troubleshooting

Q: Encounter an `unknown error` in runtime

A: Run the following command

```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```
