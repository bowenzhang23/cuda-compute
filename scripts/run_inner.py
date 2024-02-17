import numpy as np
import cuda_compute_wrapper as c2w
from time import perf_counter_ns

length = 10_000_000

x = np.random.rand(length)
y = np.random.rand(length)
vx = c2w.Vectorf(x.tolist(), length)
vy = c2w.Vectorf(y.tolist(), length)

start = perf_counter_ns()
xy = np.dot(x, y)
end = perf_counter_ns()
print(f"Inner = {xy:.4f}, Duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
xy = c2w.inner(vx, vy)
end = perf_counter_ns()
print(f"Inner = {xy:.4f}, Duration = {(end - start) * 1e-6:.4f} ms")
