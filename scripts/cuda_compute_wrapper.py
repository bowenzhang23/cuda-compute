import cuda_compute


class Vectorf(cuda_compute.VectorfBase):
    pass


class Vectori(cuda_compute.VectoriBase):
    pass


class Matrixf(cuda_compute.MatrixfBase):
    pass


class Matrixi(cuda_compute.MatrixiBase):
    pass


def device_query():
    cuda_compute.device_query()


def sgemm(a, b):
    return cuda_compute.sgemm(a, b)


def igemm(a, b):
    return cuda_compute.igemm(a, b)
