from libc.math cimport sqrt, cos, sin, M_PI
import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_factorials(int max_index, double[:] factorials):
    factorials[0] = 1.0
    cdef int i
    for i in range(1, max_index + 1):
        factorials[i] = i * factorials[i - 1]
        

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_odd_factorials(int max_index, double[:] odd_factorials):
    odd_factorials[0] = 1.0
    cdef int i
    for i in range(1, max_index + 1):
        if (i % 2 == 1):
            odd_factorials[i] = i * odd_factorials[i - 1]
        else:
            odd_factorials[i] = odd_factorials[i - 1]

@cython.boundscheck(False)
@cython.wraparound(False)            
cdef compute_lejandr(double x, int l_max, double[:] odd_factorials, double[:, :] lejandr):
    cdef int l, m
    cdef double now, multiplier
    
    now = 1.0
    multiplier = sqrt(1.0 - x * x)
    
    
    for m in range(0, l_max + 1):
        lejandr[m, m] = odd_factorials[2 * m - 1] * now
        now = -1 * multiplier * now
    lejandr[0, 0] = 1.0
    
    
    for m in range(l_max):
        lejandr[m + 1, m] = x * (2 * m + 1) * lejandr[m, m]
        
    for m in range(l_max + 1):
        for l in range(m + 2, l_max + 1):
            lejandr[l, m] = (x * (2 * l - 1) * lejandr[l - 1, m] - (l + m - 1) * lejandr[l - 2, m]) / (l - m)

            
@cython.boundscheck(False)
@cython.wraparound(False)
cdef compute_spherical_coefficients_single(double theta, double fi, int l_max, 
                                   double[:] odd_factorials, double[:] factorials,
                                   double[:, :] lejandr_placeholder, double[:, :] ans_placeholder):
    cdef double x = cos(theta)
    compute_lejandr(x, l_max, odd_factorials, lejandr_placeholder)
    cdef int l, m
    cdef double amplitude
    for l in range(l_max + 1):
        
        ans_placeholder[l, l] = sqrt((2 * l + 1) / (2.0 * 4.0 * M_PI)) * lejandr_placeholder[l, 0]
        
        for m in range(1, l + 1):
            amplitude = sqrt((2 * l + 1) / (4.0 * M_PI) * factorials[l - m] / factorials[l + m]) * lejandr_placeholder[l, m]
            
            ans_placeholder[l, m + l] = amplitude * cos(m * fi)
            ans_placeholder[l, -m + l] = amplitude * sin(m * fi)
           
               
            

@cython.boundscheck(False)
@cython.wraparound(False)                
cpdef compute_spherical_coefficients(double[:] theta, double[:] fi, int l_max,
                                     double[:] odd_factorials, double[:] factorials,
                                     double[:, :] lejandr_placeholder, double[:, :, :] ans_placeholder):
    cdef int i
    for i in range(theta.shape[0]):
        compute_spherical_coefficients_single(theta[i], fi[i], l_max, 
                                              odd_factorials, factorials,
                                              lejandr_placeholder, ans_placeholder[i, :, :])

        
class SphericalHarmonicsCalculator:
    def _reserve(self, max_order):
        self.factorials = np.empty([2 * max_order + 1], dtype = np.float)
        self.odd_factorials = np.empty([2 * max_order + 1], dtype = np.float)
        compute_factorials(2 * max_order, self.factorials)
        compute_odd_factorials(2 * max_order, self.odd_factorials)
        self.lejandr_placeholder = np.empty([max_order + 1, max_order + 1], dtype = np.float)
        self.max_order_ = max_order
        
    def __init__(self, max_order = 0):
        self._reserve(max_order)
    
    def compute(self, theta, fi, l_max):
        theta = np.asarray(theta)
        fi = np.asarray(fi)
        if (l_max > self.max_order_):
            self._reserve(l_max * 2)
        result = np.empty([theta.shape[0], l_max + 1, 2 * l_max + 1])
        compute_spherical_coefficients(theta, fi, l_max, self.odd_factorials, self.factorials, 
                                       self.lejandr_placeholder, result)
        return result
        
    