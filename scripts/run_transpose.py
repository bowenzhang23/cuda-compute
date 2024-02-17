import numpy as np
import cuda_compute_wrapper as c2w
from time import perf_counter_ns

m = 4000
n = 4000

x = np.arange(m * n, dtype=np.float32)
mx = c2w.Matrixf(x.tolist(), m, n)

x = x.reshape(m, -1)

start = perf_counter_ns()
x = np.transpose(x).copy()
end = perf_counter_ns()
print(f"Duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
mx = mx.transpose()
end = perf_counter_ns()
print(f"Duration = {(end - start) * 1e-6:.4f} ms")
