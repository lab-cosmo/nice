import numpy as np
from cython.parallel import prange
cimport cython
from multiprocessing import cpu_count

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef multiply_serial(double[:, :] x, double[:, :] m):
    result = np.empty([x.shape[0], m.shape[1]])
    cdef double[:, :] res_view = result
    cdef int i, j, k
    cdef int n_objects = x.shape[0], size_new = m.shape[1], size_old = x.shape[1]
    for i in range(n_objects):
        for j in range(size_new):
            res_view[i, j] = 0.0
            for k in range(size_old):
                res_view[i, j] += x[i, k] * m[k, j]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef multiply_parallel_static(double[:, :] x, double[:, :] m, num_threads = None):
    cdef int num_threads_int
    if (num_threads is not None):
        num_threads_int = num_threads
    else:
        num_threads_int = cpu_count()
    result = np.empty([x.shape[0], m.shape[1]])
    cdef double[:, :] res_view = result
    cdef int i, j, k
    cdef int n_objects = x.shape[0], size_new = m.shape[1], size_old = x.shape[1]
    for i in prange(n_objects, schedule = "static", nogil = True, num_threads = num_threads_int):
        for j in range(size_new):
            res_view[i, j] = 0.0
            for k in range(size_old):
                res_view[i, j] += x[i, k] * m[k, j]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef multiply_parallel_default(double[:, :] x, double[:, :] m):
    result = np.empty([x.shape[0], m.shape[1]])
    cdef double[:, :] res_view = result
    cdef int i, j, k
    cdef int n_objects = x.shape[0], size_new = m.shape[1], size_old = x.shape[1]
    for i in prange(n_objects, nogil = True):
        for j in range(size_new):
            res_view[i, j] = 0.0
            for k in range(size_old):
                res_view[i, j] += x[i, k] * m[k, j]
    return result