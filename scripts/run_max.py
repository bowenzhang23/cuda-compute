import numpy as np
import cuda_compute_wrapper as c2w
from time import perf_counter_ns

length = 10_000_000

x = np.random.rand(length)
vx = c2w.Vectorf(x.tolist(), length)

start = perf_counter_ns()
val = np.max(x)
end = perf_counter_ns()
print(f"Max = {val:.4f}, Duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
val, idx = vx.max()
end = perf_counter_ns()
print(f"Max = {val:.4f}, Idx = {idx}, Duration = {(end - start) * 1e-6:.4f} ms")

