import numpy as np
import cuda_compute_wrapper as c2w
from time import perf_counter_ns

length = 1_000_000

x = np.random.rand(length)
vx = c2w.Vectorf(x.tolist(), length)

start = perf_counter_ns()
np.sort(x)
end = perf_counter_ns()
print(f"Duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
vx.sort()
end = perf_counter_ns()
print(f"Duration = {(end - start) * 1e-6:.4f} ms")
