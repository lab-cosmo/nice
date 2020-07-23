from libc.math cimport sin, M_PI
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef simple_sin(int N, double[:] rs, double r_cut):
    ans = np.empty([rs.shape[0], N], dtype = np.float)
    cdef double[:, :] ans_view = ans
    cdef int i, j
    for i in range(rs.shape[0]):
        for j in range(N):
            ans_view[i][j] = sin(rs[i] / r_cut * (j + 1) * M_PI)
    return ans