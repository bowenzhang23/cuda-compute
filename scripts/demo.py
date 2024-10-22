import cuda_compute_wrapper as c2w

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

d = c2w.Vectori([5, 7, 3, 1, -1, -5, -6, 0, 4], 9)
d.sort_()
print(f"{d.cpu()=}")
d.sort_(False)
print(f"{d.cpu()=}")
print(f"{d.sum()=}")
print(f"{type(d), type(d.into(3))=}")
print(f"{d.into(3).shape()=}")

e = c2w.Vectori([-5, 7, 9, 3, -2, 0, -1, 10, -12], 9)
e = e.sorted()
print(f"{e.cpu()=}")
e = e.sorted(False)
print(f"{e.cpu()=}")

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

print(f"{a.transpose().cpu()=}")
print(f"{(c2w.gemm(a, b)).cpu()=}")
print(f"{type(c), type(c.into())=}")
print(f"{c.row(1).cpu()=}")
print(f"{c.col(1).cpu()=}")
print(f"{c.shape(), c.reshape_(1), c.shape()=}")

a = c2w.Matrixf([1, 2, 3, 4], 2, 2)
b = c2w.Vectorf([1, 2], 2)
print(f"{c2w.inner(a, b).cpu()=}")
print(f"{c2w.inner(b, a).cpu()=}")
print(f"{a.cpu()=}")
print(f"{a.max()=}")
print(f"{a.max(1).cpu()=}")
print(f"{a.min()=}")
print(f"{a.min(1).cpu()=}")
print(f"{a.into().cpu()=}")
print(f"{a.into().sum()=}")
print(f"{a.sum()=}")
print(f"{b.sum()=}")
print(f"{a.sum(1).cpu()=}")
print(f"{a.into().mean()=}")
print(f"{a.mean()=}")
print(f"{a.mean(1).cpu()=}")
