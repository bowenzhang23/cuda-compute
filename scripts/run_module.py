import cuda_compute_wrapper as c2w
from time import perf_counter_ns
import numpy as np

print(c2w.device_query())

a = c2w.Vectorf([1, 2, 3, 4, 5], 5)
b = c2w.Vectorf([1, 2, 3, 4, 5], 5)
c = c2w.Vectorf([5, 4, 3, 2, 1], 5)

print(f"{(a == 3).cpu()=}")
print(f"{(a != 3).cpu()=}")
print(f"{(3 == a).cpu()=}")
print(f"{(3 != a).cpu()=}")
print(f"{(a > 3).cpu()=}")
print(f"{(a >= 3).cpu()=}")
print(f"{(a < 3).cpu()=}")
print(f"{(a <= 3).cpu()=}")
print(f"{(3 > a).cpu()=}")
print(f"{(3 >= a).cpu()=}")
print(f"{(3 < a).cpu()=}")
print(f"{(3 <= a).cpu()=}")
print(f"{(a + b).cpu()=}")
print(f"{(a - b).cpu()=}")
print(f"{(a * b).cpu()=}")
print(f"{(a / b).cpu()=}")
print(f"{(2 + b).cpu()=}")
print(f"{(a + 2).cpu()=}")
print(f"{(2 - b).cpu()=}")
print(f"{(a - 2).cpu()=}")
print(f"{(2 * b).cpu()=}")
print(f"{(a * 2).cpu()=}")
print(f"{(2 / b).cpu()=}")
print(f"{(a / 2).cpu()=}")
print(f"{c2w.max(a, c).cpu()=}")
print(f"{c2w.min(a, c).cpu()=}")
print(f"{c2w.max(a, 3).cpu()=}")
print(f"{c2w.max(3, a).cpu()=}")
print(f"{c2w.min(a, 3).cpu()=}")
print(f"{c2w.min(3, a).cpu()=}")
print(f"{c2w.inner(a, b)=}")
print(f"{c2w.inner(a, c)=}")
print(f"{c2w.distance(a, c)=}")
print(f"{c2w.mod(a)=}")
print(f"{c2w.mod2(a)=}")


a = c2w.Matrixf([1, 2, 3, 4], 2, 2)
b = c2w.Matrixf([1, 2, 3, 4], 2, 2)
c = c2w.Matrixf([4, 3, 2, 1], 2, 2)

print(f"{(a == 3).cpu()=}")
print(f"{(a != 3).cpu()=}")
print(f"{(3 == a).cpu()=}")
print(f"{(3 != a).cpu()=}")
print(f"{(a > 3).cpu()=}")
print(f"{(a >= 3).cpu()=}")
print(f"{(a < 3).cpu()=}")
print(f"{(a <= 3).cpu()=}")
print(f"{(3 > a).cpu()=}")
print(f"{(3 >= a).cpu()=}")
print(f"{(3 < a).cpu()=}")
print(f"{(3 <= a).cpu()=}")
print(f"{(a + b).cpu()=}")
print(f"{(a - b).cpu()=}")
print(f"{(a * b).cpu()=}")
print(f"{(a / b).cpu()=}")
print(f"{(2 + b).cpu()=}")
print(f"{(a + 2).cpu()=}")
print(f"{(2 - b).cpu()=}")
print(f"{(a - 2).cpu()=}")
print(f"{(2 * b).cpu()=}")
print(f"{(a * 2).cpu()=}")
print(f"{(2 / b).cpu()=}")
print(f"{(a / 2).cpu()=}")
print(f"{c2w.max(a, c).cpu()=}")
print(f"{c2w.min(a, c).cpu()=}")
print(f"{c2w.max(a, 3).cpu()=}")
print(f"{c2w.max(3, a).cpu()=}")
print(f"{c2w.min(a, 3).cpu()=}")
print(f"{c2w.min(3, a).cpu()=}")

print(a.transpose().cpu())
print((c2w.gemm(a, b)).cpu())

m = 1024
k = 4096
n = 2048
va = np.arange(m * k, dtype=np.float32)
vb = np.arange(k * n, dtype=np.float32)

a = va.tolist()
b = vb.tolist()

print("transfer")
a = c2w.Matrixf(a, m, k)
b = c2w.Matrixf(b, k, n)
print("finished")

start = perf_counter_ns()
c = c2w.gemm(a, b)
end = perf_counter_ns()
print(f"matmul duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
d = -c
end = perf_counter_ns()
print(f"copy duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
d = c.transpose()
end = perf_counter_ns()
print(f"transpose duration = {(end - start) * 1e-6:.4f} ms")

a = va.reshape(m, -1)
b = vb.reshape(k, -1)

start = perf_counter_ns()
c = np.matmul(a, b)
end = perf_counter_ns()
print(f"matmul duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
d = c.transpose().copy()
end = perf_counter_ns()
print(f"transpose duration = {(end - start) * 1e-6:.4f} ms")

va = np.arange(1 << 24, dtype=np.float32)
vb = np.arange(1 << 24, dtype=np.float32)

a = va.tolist()
b = vb.tolist()

print("transfer")
a = c2w.Vectorf(a, len(a))
b = c2w.Vectorf(b, len(b))
print("finished")

start = perf_counter_ns()
c = c2w.inner(a, b)
end = perf_counter_ns()
print(f"inner duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
c = np.dot(va, vb)
end = perf_counter_ns()
print(f"inner duration = {(end - start) * 1e-6:.4f} ms")

va = np.random.rand(1 << 20)
a = va.tolist()
a = c2w.Vectorf(a, len(a))

start = perf_counter_ns()
val, idx = a.max()
end = perf_counter_ns()
print(f"max {val:.6f} {idx} duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
c = np.max(va)
end = perf_counter_ns()
print(f"max {c:.6f} duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
c = a.mean()
end = perf_counter_ns()
print(f"mean {c:.6f} duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
c = np.mean(va)
end = perf_counter_ns()
print(f"mean {c:.6f} duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
c = +a
end = perf_counter_ns()
print(f"copy duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
c = a.reversed()
end = perf_counter_ns()
print(f"reversed duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
vc = va[::-1].copy()
end = perf_counter_ns()
print(f"reversed duration = {(end - start) * 1e-6:.4f} ms")
