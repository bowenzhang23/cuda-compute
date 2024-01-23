from cuda_compute import *
from time import perf_counter_ns
import numpy as np

a = VectorfBase([1, 2, 3, 4, 5], 5)
b = VectorfBase([1, 2, 3, 4, 5], 5)

print((a + b).cpu())
print((a - b).cpu())
print((2 * b).cpu())
print((a * 2).cpu())

a = MatrixfBase([1, 2, 3, 4], 2, 2)
b = MatrixfBase([1, 2, 3, 4], 2, 2)

print((a + b).cpu())
print((a - b).cpu())
print((2 * b).cpu())
print((a * 2).cpu())

print(a.transpose().cpu())
print((sgemm(a, b)).cpu())

n = 256

va = np.arange(n * n, dtype=np.float32)
vb = np.arange(n * n, dtype=np.float32)

a = va.tolist()
b = vb.tolist()

print("transfer")
a = MatrixfBase(a, n, n)
b = MatrixfBase(b, n, n)
print("finished")

start = perf_counter_ns()
c = sgemm(a, b)
end = perf_counter_ns()
print(f"matmul duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
d = c.transpose()
end = perf_counter_ns()
print(f"transpose duration = {(end - start) * 1e-6:.4f} ms")

a = va.reshape(n, -1)
b = vb.reshape(n, -1)

start = perf_counter_ns()
c = np.matmul(a, b)
end = perf_counter_ns()
print(f"matmul duration = {(end - start) * 1e-6:.4f} ms")

start = perf_counter_ns()
d = c.transpose()
end = perf_counter_ns()
print(f"transpose duration = {(end - start) * 1e-6:.4f} ms")