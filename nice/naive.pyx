import numpy as np
cimport cython
from nice_utilities cimport single_contraction, min_c, abs_c


'''@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_bispectrum_single(double[:, :, :, :, :] clebsh_gordan, double[:] c1,
                   double[:] c2, double[:] c3, int l1, int l2, int l, double[:, :] buff):
    
    
    single_contraction(clebsh_gordan, c1, l1, c2, l2, l, buff[0], buff[1:])
    return compute_powerspectrum_single(buff[0], c3, l) 
    

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] compute_bispectrum(double[:, :, :, :, :] clebsh_gordan, double[:, :, :] c, 
                         int l_max, bint only_even = False):
    buffer = np.empty([5, 2 * l_max + 1])
    cdef double[:, :] buff = buffer
    cdef int n_radial = c.shape[0]
    cdef int ans_size = 0
    cdef int n1, n2, n3, l1, l2, l3
    for n1 in range(n_radial):
        for n2 in range(n1, n_radial):
            for n3 in range(n2, n_radial):
                for l1 in range(l_max + 1):
                    for l2 in range(l_max + 1):
                        for l3 in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
                            if (only_even and ((l1 + l2 + l3) % 2 == 1)):
                                continue
                            ans_size += 1
    ans = np.zeros([ans_size])
    
    cdef double[:] ans_view = ans
    cdef int now = 0
    for n1 in range(n_radial):
        for n2 in range(n1, n_radial):
            for n3 in range(n2, n_radial):
                for l1 in range(l_max + 1):
                    for l2 in range(l_max + 1):
                        for l3 in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
                            if (only_even and ((l1 + l2 + l3) % 2 == 1)):
                                continue
                            ans_view[now] = compute_bispectrum_single(clebsh_gordan,
                                                                 c[n1, l1, :], c[n2, l2, :], c[n3, l3, :],
                                                                 l1, l2, l3, buff)
                            now += 1
    return ans'''
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_powerspectrum_single(double[:] c1, double[ :] c2, int l):
    cdef int m1, m2, m
    cdef double result = 0.0
    for m in range(-l, l + 1):
        now = c1[m + l] * c2[m + l]
        result += now
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] compute_powerspectrum(double[:, :, :] c, int l_max):
    cdef int n_radial = c.shape[0]
    cdef int ans_size = 0
    cdef int n1, n2, l
    for n1 in range(n_radial):
        for n2 in range(n1, n_radial):
            for l in range(l_max + 1):
                ans_size += 1
    ans = np.empty([ans_size])
    cdef double[:] ans_view = ans
    cdef int now = 0
    for n1 in range(n_radial):
        for n2 in range(n1, n_radial):
            for l in range(l_max + 1):
                ans_view[now] = compute_powerspectrum_single(c[n1, l, :], c[n2, l, :], l)
                now += 1
    return ans
    
    

                
    