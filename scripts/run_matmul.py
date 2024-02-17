import numpy as np
import cuda_compute_wrapper as c2w
from time import perf_counter_ns

m = 1024
k = 4096
n = 2048

x = np.arange(m * k, dtype=np.float32)
y = np.arange(k * n, dtype=np.float32)

mx = c2w.Matrixf(x.tolist(), m, k)
my = c2w.Matrixf(y.tolist(), k, n)

x = x.reshape(m, -1)
y = y.reshape(k, -1)

start = perf_counter_ns()
z = np.matmul(x, y)
end = perf_counter_ns()
print(f"Duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
mz = c2w.gemm(mx, my)
end = perf_counter_ns()
print(f"Duration = {(end - start) * 1e-6:.4f} ms")
