cimport cython
import numpy as np
from libc.math cimport fabs


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pack_dense(double[:, :, :] covariants, int l,
                 int n_feat, int desired_n_feat):    
    cdef int n_envs = covariants.shape[0]
    cdef int num_per_feat = (l + 1)
    res = np.zeros([n_envs * (2 * l + 1), desired_n_feat])
    cdef double[:, :] res_view = res
    cdef int env_ind, feat_ind, now, m
    
    
    for feat_ind in range(n_feat):
        now = 0
        for env_ind in range(n_envs):           
            for m in range(2 * l + 1):
                res_view[now, feat_ind] = covariants[env_ind, feat_ind, m]
                now += 1
                    
    return res
    
@cython.boundscheck(False)
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
    return res
                
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unpack_dense(double[:, :] packed, int n_envs, int l, int n_feat):
   
    
    res = np.zeros([n_envs, n_feat, 2 * l + 1])
    cdef double[:, :, :] res_view = res
    cdef int feat_ind, now, env_ind, m
    
    for feat_ind in range(n_feat):
        now = 0
        for env_ind in range(n_envs):           
            for m in range(2 * l + 1):
                res_view[env_ind, feat_ind, m] = packed[now, feat_ind]
                now += 1
    return res



    
from sklearn.decomposition import TruncatedSVD #not center the data
class UnrollingIndividualPCA(TruncatedSVD):
    def __init__(self, *args, normalize_importances = True, **kwargs):
        self.normalize_importances_ = normalize_importances
        super().__init__(*args, **kwargs)     
        
    def fit(self, *args):
        if (len(args) == 1):
            return super().fit(args[0])
        #print("num components: ", self.n_components)
        covariants, n_feat, l = args
        if (self.n_components > n_feat):
            #print("in if: ", self.n_components, n_feat)
            self.n_components = n_feat           
        
        self.l_ = l
        if (self.n_components < n_feat):
            packed = pack_dense(covariants, l, n_feat, n_feat)
        if (self.n_components == n_feat):
            packed = pack_dense(covariants, l, n_feat, n_feat + 1)
        res = super().fit_transform(packed)
        
        self.importances_ = np.mean(res * res, axis = 0)
        if (self.normalize_importances_):
            self.importances_ = self.importances_ / np.sum(self.importances_)
        indices = np.argsort(self.importances_)[::-1]
        self.importances_ = self.importances_[indices]
        self.components_ = self.components_[indices]
    
    def fit_transform(self, *args):
        if (len(args) ==1):
            return super().fit_transform(args[0])
        covariants, n_feat, l = args
        #print("num components: ", self.n_components)
        if (self.n_components > n_feat):
            #print("in if: ", self.n_components, n_feat)
            self.n_components = n_feat           
        
        self.l_ = l
        if (self.n_components < n_feat):
            packed = pack_dense(covariants, l, n_feat, n_feat)
        if (self.n_components == n_feat):
            packed = pack_dense(covariants, l, n_feat, n_feat + 1)   
            
        res = super().fit_transform(packed)        
        self.importances_ = np.mean(res * res, axis = 0)
        if (self.normalize_importances_):
            self.importances_ = self.importances_ / np.sum(self.importances_)
        indices = np.argsort(self.importances_)[::-1]
        self.importances_ = self.importances_[indices]
        self.components_ = self.components_[indices]
        
        res = super().transform(packed)
        return unpack_dense(res, covariants.shape[0],
                                         self.l_, self.n_components)
    
        
    def transform(self, *args, method = 'serial'):
        
        
        if (len(args) == 1):
            return super().transform(args)
        #print("components shape: ", self.components_.shape)
        #print("num components: ", self.n_components)
        covariants, n_feat, l = args
        if (method == 'serial'):            
            return transform_inplace(covariants, self.components_, 
                            l, n_feat)
        else:
            if (self.n_components < n_feat):
                packed = pack_dense(covariants, l, n_feat, n_feat)
            if (self.n_components == n_feat):
                packed = pack_dense(covariants, l, n_feat, n_feat + 1)    
            res = super().transform(packed)
            return unpack_dense(res, covariants.shape[0],
                                         self.l_, self.n_components)

 
       
    
        
        