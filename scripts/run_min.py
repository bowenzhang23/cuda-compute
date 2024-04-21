import numpy as np
import cuda_compute_wrapper as c2w
from time import perf_counter_ns

length = 10_000_000

x = np.random.rand(length) - 0.5
vx = c2w.Vectorf(x.tolist(), length)

start = perf_counter_ns()
val = np.min(x)
end = perf_counter_ns()
print(f"Min = {val:.4f}, Duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
val, idx = vx.min()
end = perf_counter_ns()
print(f"Min = {val:.4f}, Idx = {idx}, Duration = {(end - start) * 1e-6:.4f} ms")

