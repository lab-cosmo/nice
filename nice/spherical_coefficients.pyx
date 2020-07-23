from libc.math cimport sin, M_PI
cimport cython
import numpy as np
from spherical_harmonics import SphericalHarmonicsCalculator

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef accumulate(double[:, :, :, :] contributions, int [:] central_indices, int[:] neighbor_indices, 
                 double[:, :, :, :] ans, int n_types):
    cdef num = central_indices.shape[0]
    cdef int n_radial = contributions.shape[1]
    cdef int l_max = contributions.shape[2] - 1
    cdef double[:, :, :] destination, source
    cdef int i, n, l, m
    cdef int shift_now
    
    for i in range(num):
        destination = ans[central_indices[i]]
        source = contributions[i]
        shift_now = neighbor_indices[i] * n_types
        for n in range(n_radial):
            for l in range(l_max + 1):
                for m in range(0, 2 * l + 1):
                    destination[n + shift_now, l, m] = destination[n + shift_now, l, m] + source[n, l, m]
                    
                    
def transform_cart_spherical(positions):
    rs = np.sqrt(np.sum(positions * positions, axis = -1))
    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]
    fi = np.arctan2(ys, xs)
    theta = np.arctan2(np.sqrt(xs * xs + ys * ys), zs)
    return rs, theta, fi


from ase.neighborlist import neighbor_list
class SphericalCoefficientsCalculator:
    def __init__(self, n_max, l_max, r_cut, radial_basis):
        self.calculator_ = SphericalHarmonicsCalculator()
        self.r_cut_ = r_cut
        self.radial_basis_ = radial_basis
        self.n_max_ = n_max
        self.l_max_ = l_max
        
    def transform_single(self, neighbors, n_atoms, n_types):
        
        central_indices, neighbor_indices, positions = neighbors
        rs, theta, fi = transform_cart_spherical(positions)
        weights = self.radial_basis_(self.n_max_, rs, self.r_cut_)


        sphericals = self.calculator_.compute(theta, fi, self.l_max_)
        ar = sphericals[:, np.newaxis, :, :] * weights[:, :, np.newaxis, np.newaxis]
        coefficients = np.zeros([n_atoms, self.n_max_ * n_types, self.l_max_ + 1, 2 * self.l_max_ + 1])
        accumulate(ar, central_indices.astype(np.int32),
                   neighbor_indices.astype(np.int32), coefficients, n_types)
        return coefficients
    
    
    