cimport cython
import numpy as np
from nice.packing import pack_dense, unpack_dense

from sklearn.decomposition import TruncatedSVD #not center the data
class UnrollingIndividualPCA(TruncatedSVD):
    def __init__(self, *args, normalize_importances = True, **kwargs):
        self.normalize_importances_ = normalize_importances
        super().__init__(*args, **kwargs)     
        
    def fit(self, *args):
        if (len(args) == 1):
            return super().fit(args[0])
        #print("num components: ", self.n_components)
        covariants, l = args
        n_feat = covariants.shape[1]
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
        self.explained_variance_ = self.explained_variance_[indices]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[indices]
        self.singular_values_ = self.singular_values_[indices]
    
    def fit_transform(self, *args):
        if (len(args) ==1):
            return super().fit_transform(args[0])
        covariants, l = args
        n_feat = covariants.shape[1]
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
    
        
    def transform(self, *args):
        
        
        if (len(args) == 1):
            return super().transform(args)
        #print("components shape: ", self.components_.shape)
        #print("num components: ", self.n_components)
        covariants, l = args
        n_feat = covariants.shape[1]
        
        if (self.n_components < n_feat):
            packed = pack_dense(covariants, l, n_feat, n_feat)
        if (self.n_components == n_feat):
            packed = pack_dense(covariants, l, n_feat, n_feat + 1)    
        res = super().transform(packed)
        return unpack_dense(res, covariants.shape[0],
                                     self.l_, self.n_components)

 
       
    
        
        