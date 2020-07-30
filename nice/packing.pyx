cimport cython
import numpy as np
from cython.parallel import prange
from multiprocessing import cpu_count

                     
                     
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pack_dense(double[:, :, :] covariants, int l,
                 int n_feat, int desired_n_feat, num_threads = None):
    cdef int num_threads_int
    if (num_threads is None):
        num_threads_int = cpu_count()
    else:
        num_threads_int = num_threads
    
    cdef int n_envs = covariants.shape[0]
    cdef int num_per_feat = (l + 1)
    res = np.empty([n_envs * (2 * l + 1), desired_n_feat])
    cdef double[:, :] res_view = res
    cdef int env_ind, feat_ind, now, m
    
    for env_ind in prange(n_envs, nogil = True, schedule = 'static', num_threads = num_threads_int):
        for feat_ind in range(n_feat):
            for m in range(2 * l + 1):
                res_view[m + env_ind * (2 * l + 1), feat_ind] = covariants[env_ind, feat_ind, m]
                
    for env_ind in prange(n_envs, nogil = True, schedule = 'static', num_threads = num_threads_int):
        for feat_ind in range(n_feat, desired_n_feat):
            for m in range(2 * l + 1):
                res_view[m + env_ind * (2 * l + 1), feat_ind] = 0.0
        
    '''for feat_ind in prange(n_feat, nogil = True, schedule = 'static', num_threads = num_threads_int):
        now = 0
        for env_ind in range(n_envs):           
            for m in range(2 * l + 1):
                res_view[now, feat_ind] = covariants[env_ind, feat_ind, m]
                now = now + 1
                
    for feat_ind in prange(n_feat, desired_n_feat, nogil = True, schedule = 'static', num_threads = num_threads_int):
        now = 0
        for env_ind in range(n_envs):           
            for m in range(2 * l + 1):
                res_view[now, feat_ind] = 0.0
                now = now + 1'''
                    
    return res
    
'''@cython.boundscheck(False)
@cython.wraparound(False)
cdef transform_inplace(double[:, :, :] covariants, double[:, :] components, 
                        int l, int n_feat):
    cdef int n_envs = covariants.shape[0]
    res = np.zeros([n_envs, components.shape[0], 2 * l + 1])
    cdef double[:, :, :] res_view = res
    cdef int feat_ind, env_ind, m, i
    
    for env_ind in range(n_envs):
        for feat_ind in range(components.shape[0]):
            for m in range(2 * l + 1):
                for i in range(n_feat):
                    res_view[env_ind, feat_ind, m] += components[feat_ind, i] * covariants[env_ind, i, m]
    return res'''
                
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unpack_dense(double[:, :] packed, int n_envs, int l, int n_feat, num_threads = None):
    cdef int num_threads_int
    if (num_threads is None):
        num_threads_int = cpu_count()
    else:
        num_threads_int = num_threads
        
    res = np.empty([n_envs, n_feat, 2 * l + 1])
    cdef double[:, :, :] res_view = res
    cdef int feat_ind, now, env_ind, m
    
    '''for feat_ind in prange(n_feat, nogil = True, schedule = 'static', num_threads = num_threads_int):
        now = 0
        for env_ind in range(n_envs):           
            for m in range(2 * l + 1):
                res_view[env_ind, feat_ind, m] = packed[now, feat_ind]
                now = now + 1'''
    
    for env_ind in prange(n_envs, nogil = True, schedule = 'static', num_threads = num_threads_int):
        for feat_ind in range(n_feat):
            for m in range(2 * l + 1):
                res_view[env_ind, feat_ind, m] = packed[m + env_ind * (2 * l + 1), feat_ind]
    return res
