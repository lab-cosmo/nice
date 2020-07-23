import numpy as np
cimport cython
from nice_utilities cimport single_contraction, min_c, abs_c, max_c
from naive cimport compute_powerspectrum
from libc.math cimport sin, M_PI, sqrt, fmax
import tqdm
import rascal
from rascal.representations import SphericalInvariants as SOAP
from rascal.representations import SphericalExpansion as SPH
from rascal.neighbourlist.structure_manager import (
        mask_center_atoms_by_species, mask_center_atoms_by_id)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef convert_rascal_coefficients(double[:, :] coefficients, int n_max, int n_types, int l_max):
    cdef int n_envs = coefficients.shape[0]
    cdef int env_ind, n, l, m
    cdef int n_radial = n_max * n_types
    cdef int now
    ans = np.zeros([n_envs, n_radial, l_max + 1, 2 * l_max + 1])
    cdef double[:, :, :, :] ans_view = ans
    
    for env_ind in range(n_envs):
        now = 0
        for n in range(n_radial):
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    ans_view[env_ind, n, l, m + l] = coefficients[env_ind, now]
                    now += 1
    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef normalize_by_ps(double[:, :, :, :] coefficients):
    cdef int n_envs = coefficients.shape[0]
    cdef int n_radial = coefficients.shape[1]
    cdef int l_max = coefficients.shape[2] - 1
    cdef int env_ind
    cdef int i, n, l, m
    cdef double self_kernel 
    cdef double multiplier
    
    for env_ind in range(n_envs):
        ps_now = compute_powerspectrum(coefficients[env_ind], l_max)
        self_kernel = 0.0
        for i in range(ps_now.shape[0]):
            self_kernel += ps_now[i] * ps_now[i]        
        multiplier = sqrt(sqrt(self_kernel))        
        for n in range(n_radial):
            for l in range(l_max + 1):
                for m in range(2 * l + 1):
                    coefficients[env_ind, n, l, m] /= multiplier
                    
                    
def get_rascal_coefficients(structures, HYPERS, n_types, normalize = True):
   
    
    sph = SPH(**HYPERS)
    n_max = HYPERS['max_radial']
    l_max = HYPERS['max_angular']
    feat = sph.transform(structures).get_features(sph)
    res = convert_rascal_coefficients(feat, n_max, n_types, l_max)
    
    if (normalize):
        normalize_by_ps(res)
    return np.array(res)



def get_rascal_coefficients_parallelized(p, structures, hypers, n_types,
                                         normalize = True, task_size = 100):
    
    tasks = []
    for i in range(0, len(structures), task_size):
        tasks.append([structures[i : i + task_size], hypers, n_types, normalize])
        
    def wrapped(task):
        return get_rascal_coefficients(*task)
    
    result = [res for res in tqdm.tqdm(p.imap(wrapped, tasks), total = len(tasks))]
    return np.concatenate(result, axis = 0)
    
            