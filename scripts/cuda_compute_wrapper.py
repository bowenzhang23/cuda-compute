import cuda_compute


class Device(cuda_compute.DeviceBase):
    pass


class Vectorf(cuda_compute.VectorfBase):
    pass


class Vectori(cuda_compute.VectoriBase):
    pass


class Matrixf(cuda_compute.MatrixfBase):
    pass


class Matrixi(cuda_compute.MatrixiBase):
    pass


def use_device(id):
    cuda_compute.use_device(id)


def current_device():
    return cuda_compute.current_device()


def device_query():
    cuda_compute.device_query()


def timer_start():
    cuda_compute.timer_start()


def timer_stop():
    return cuda_compute.timer_stop()


def max(a, b):
    return cuda_compute.max(a, b)


def min(a, b):
    return cuda_compute.min(a, b)


def gemm(a, b):
    return cuda_compute.gemm(a, b)


def inner(a, b):
    return cuda_compute.inner(a, b)
