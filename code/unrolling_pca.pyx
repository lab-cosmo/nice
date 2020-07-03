cimport cython
import numpy as np
from libc.math cimport fabs

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pack_dense(double[:, :, :, :] coefficients):
    cdef int n_envs = coefficients.shape[0]
    cdef int n_feat = coefficients.shape[1]   
    cdef int l_max = coefficients.shape[2] - 1
    
    
   
    cdef int num_per_feat = (l_max + 1) * (l_max + 1)
    res = np.zeros([num_per_feat * n_envs, n_feat])
    cdef double[:, :] res_view = res
    
    cdef int env_ind, feat_ind, now, l, m
    for feat_ind in range(n_feat):
        now = 0
        for env_ind in range(n_envs):
            for l in range(l_max + 1):
                for m in range(2 * l + 1):
                    res_view[now, feat_ind] = coefficients[env_ind, feat_ind, l, m]
                    now += 1
                    
    return res

cpdef unpack_dense(double[:, :] packed, int n_envs, int l_max):
    cdef int n_feat = packed.shape[1]
    
    res = np.zeros([n_envs, n_feat, l_max + 1, 2 * l_max + 1])
    cdef double[:, :, :, :] res_view = res
    cdef int feat_ind, now, env_ind, l, m
    
    for feat_ind in range(n_feat):
        now = 0
        for env_ind in range(n_envs):
            for l in range(l_max + 1):
                for m in range(2 * l + 1):
                    res_view[env_ind, feat_ind, l, m] = packed[now, feat_ind]
                    now += 1
    return res

cpdef get_signs(double[:, :] ar, epsilon = 1e-10):
    res = np.zeros([ar.shape[0]])
    cdef double[:] res_view = res
    cdef int n_feat = ar.shape[1]
    cdef int i, j
    cdef double max_absolute_now 
    for i in range(ar.shape[0]):
        max_absolute_now = ar[i, 0]
        for j in range(n_feat):
            if (fabs(ar[i, j]) > fabs(max_absolute_now)):
                max_absolute_now = ar[i, j]
                
        if (max_absolute_now > epsilon):
            res_view[i] = 1.0
        if (max_absolute_now < epsilon):
            res_view[i] = -1.0
        
    return res


from sklearn.decomposition import TruncatedSVD #not center the data
class UnrollingPCA(TruncatedSVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)     
        
    def fit_transform(self, coefficients):
        self.n_feat_ = coefficients.shape[1]
        self.l_max_ = coefficients.shape[2] - 1
        packed = pack_dense(coefficients)      
        res = super().fit_transform(packed)
        return unpack_dense(res, coefficients.shape[0],
                                         self.l_max_)
    
    def fit(self, coefficients):
        self.n_feat_ = coefficients.shape[1]
        self.l_max_ = coefficients.shape[2] - 1
        packed = pack_dense(coefficients)  
        super().fit(packed)
        
    def transform(self, coefficients):
        if (self.n_feat_ != coefficients.shape[1]):
            raise ValueError("wrong shape")
        if (self.l_max_ != coefficients.shape[2] - 1):
            raise ValueError("wrong shape")
        packed = pack_dense(coefficients)
        res = super().transform(packed)       
        return unpack_dense(res, coefficients.shape[0],
                                         self.l_max_)
        
        