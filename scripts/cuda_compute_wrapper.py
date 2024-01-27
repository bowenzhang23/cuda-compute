import cuda_compute


class Vectorf(cuda_compute.VectorfBase):
    pass


class Vectori(cuda_compute.VectoriBase):
    pass


class Matrixf(cuda_compute.MatrixfBase):
    pass


class Matrixi(cuda_compute.MatrixiBase):
    pass


def max(a, b):
    return cuda_compute.max(a, b)


def min(a, b):
    return cuda_compute.min(a, b)


def device_query():
    cuda_compute.device_query()


def gemm(a, b):
    return cuda_compute.gemm(a, b)


def inner(a, b):
    return cuda_compute.inner(a, b)
