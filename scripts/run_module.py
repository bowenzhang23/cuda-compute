from cuda_compute import *
from time import perf_counter_ns
import numpy as np

print(device_query())

a = VectorfBase([1, 2, 3, 4, 5], 5)
b = VectorfBase([1, 2, 3, 4, 5], 5)

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

a = MatrixfBase([1, 2, 3, 4], 2, 2)
b = MatrixfBase([1, 2, 3, 4], 2, 2)

print(f"{(-a).cpu()=}")
print(f"{(+a).cpu()=}")
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

print(a.transpose().cpu())
print((sgemm(a, b)).cpu())

m = 1024
k = 4096
n = 2048
va = np.arange(m * k, dtype=np.float32)
vb = np.arange(k * n, dtype=np.float32)

a = va.tolist()
b = vb.tolist()

print("transfer")
a = MatrixfBase(a, m, k)
b = MatrixfBase(b, k, n)
print("finished")

start = perf_counter_ns()
c = sgemm(a, b)
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
d = c.transpose()
end = perf_counter_ns()
print(f"transpose duration = {(end - start) * 1e-6:.4f} ms")