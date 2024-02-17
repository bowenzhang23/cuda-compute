import numpy as np
import cuda_compute_wrapper as c2w
from time import perf_counter_ns

length = 10_000_000

x = np.random.rand(length)
vx = c2w.Vectorf(x.tolist(), length)

start = perf_counter_ns()
m = np.mean(x)
end = perf_counter_ns()
print(f"Mean = {m:.4f}, Duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
m = vx.mean()
end = perf_counter_ns()
print(f"Mean = {m:.4f}, Duration = {(end - start) * 1e-6:.4f} ms")
