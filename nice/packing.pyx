cimport cython
import os
import numpy as np
from cython.parallel import prange
from multiprocessing import cpu_count

cdef int switch_to_parallel_after = 36000000

                     
                     
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pack_dense(double[:, :, :] covariants, int l,
                 int n_feat, int desired_n_feat, num_threads = None):
    cdef int num_threads_int
    if (num_threads is None):
        num_threads_int = len(os.sched_getaffinity(0))
    else:
        num_threads_int = num_threads
    
    cdef int n_envs = covariants.shape[0]
    cdef int num_per_feat = (l + 1)
    res = np.empty([n_envs * (2 * l + 1), desired_n_feat])
    cdef double[:, :] res_view = res
    cdef int env_ind, feat_ind, now, m
    
    if (n_feat * (2 * l + 1) * n_envs) > switch_to_parallel_after:
        for env_ind in prange(n_envs, nogil = True, schedule = 'static', num_threads = num_threads_int):
            for feat_ind in range(n_feat):
                for m in range(2 * l + 1):
                    res_view[m + env_ind * (2 * l + 1), feat_ind] = covariants[env_ind, feat_ind, m]

        for env_ind in prange(n_envs, nogil = True, schedule = 'static', num_threads = num_threads_int):
            for feat_ind in range(n_feat, desired_n_feat):
                for m in range(2 * l + 1):
                    res_view[m + env_ind * (2 * l + 1), feat_ind] = 0.0
                    
    else:
        for env_ind in range(n_envs):
            for feat_ind in range(n_feat):
                for m in range(2 * l + 1):
                    res_view[m + env_ind * (2 * l + 1), feat_ind] = covariants[env_ind, feat_ind, m]

        for env_ind in range(n_envs):
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
        num_threads_int = len(os.sched_getaffinity(0))
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
    if (n_feat * (2 * l + 1) * n_envs) > switch_to_parallel_after:
        for env_ind in prange(n_envs, nogil = True, schedule = 'static', num_threads = num_threads_int):
            for feat_ind in range(n_feat):
                for m in range(2 * l + 1):
                    res_view[env_ind, feat_ind, m] = packed[m + env_ind * (2 * l + 1), feat_ind]
                    
    else:
        for env_ind in range(n_envs):
            for feat_ind in range(n_feat):
                for m in range(2 * l + 1):
                    res_view[env_ind, feat_ind, m] = packed[m + env_ind * (2 * l + 1), feat_ind]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef copy_parallel(double[:, :] source, double[:, :] destination, num_threads = None):
    cdef int num_threads_int
    if (num_threads is None):
        num_threads_int = len(os.sched_getaffinity(0))
    else:
        num_threads_int = num_threads
        
    cdef int env_ind, feat_ind
    cdef int n_feat = source.shape[1]
    if (source.shape[0] * source.shape[1] > switch_to_parallel_after):
        for env_ind in prange(source.shape[0], nogil = True, schedule = 'static', num_threads = num_threads_int):
            for feat_ind in range(n_feat):
                destination[env_ind, feat_ind] = source[env_ind, feat_ind]
    else:
        for env_ind in range(source.shape[0]):
            for feat_ind in range(n_feat):
                destination[env_ind, feat_ind] = source[env_ind, feat_ind]
    
            
def unite_parallel(blocks, num_threads = None):
    total_size = 0
    for block in blocks:
        total_size += block.shape[1]
    res = np.empty([blocks[0].shape[0], total_size])
    now = 0
    for block in blocks:
        copy_parallel(block, res[:, now : now + block.shape[1]], num_threads = num_threads)
        now += block.shape[1]
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef subtract_parallel(double[:, :] a, double[:, :] b, num_threads = None):
    result = np.empty([a.shape[0], a.shape[1]])
    cdef double[:, :] result_view = result
    
    cdef int num_threads_int
    if (num_threads is None):
        num_threads_int = len(os.sched_getaffinity(0))
    else:
        num_threads_int = num_threads
        
    cdef int env_ind, feat_ind
    cdef int n_feat = a.shape[1]
    if (a.shape[0] * a.shape[1] > switch_to_parallel_after):
        for env_ind in prange(a.shape[0], nogil = True, schedule = 'static', num_threads = num_threads_int):
            for feat_ind in range(n_feat):
                result_view[env_ind, feat_ind] = a[env_ind, feat_ind] - b[env_ind, feat_ind]
    else:
        for env_ind in range(a.shape[0]):
            for feat_ind in range(n_feat):
                result_view[env_ind, feat_ind] = a[env_ind, feat_ind] - b[env_ind, feat_ind]
            
    return result
     

'''@cython.boundscheck(False)
@cython.wraparound(False)
cpdef accumulate(double[:, :] values, int[:] structure_indices, int central_now,
                double[:, :] ans):
    
    cdef int env_ind, feat_ind, n_feat = values.shape[1]
    cdef int now = 0
    for env_ind in range(values.shape[0]):      
        for feat_ind in range(n_feat):
            ans[structure_indices[env_ind], feat_ind] += values[env_ind, feat_ind]
    
def accumulate_to_structures(structures, values):
    all_species = []
    for structure in structures:
        all_species.append(structure.get_atomic_numbers())
    all_species = np.concatenate(all_species, axis = 0)
    species = np.unique(all_species)
    all_species = all_species.astype(np.int32)
    species = species.astype(np.int32)
    
    result = {}
    for specie in tqdm.tqdm(species):
        num_now = np.sum(all_species == specie)
        result[specie] = np.empty([num_now, coefficients.shape[1], coefficients.shape[2], coefficients.shape[3]])
        copy_coefs(coefficients, all_species, specie, result[specie])
    return result   '''
