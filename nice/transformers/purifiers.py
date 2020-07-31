import numpy as np
from nice.unrolling_individual_pca import UnrollingIndividualPCA
#from cython.parallel cimport prange

from nice.thresholding import get_thresholded_tasks
from nice.nice_utilities import do_partial_expansion, Data, get_sizes
from nice.ClebschGordan import ClebschGordan, check_clebsch_gordan
from nice.packing import unite_parallel, subtract_parallel, pack_dense, unpack_dense
from parse import parse
import warnings
from sklearn.linear_model import Ridge
from sklearn.base import clone


DEFAULT_LINEAR_REGRESSOR = Ridge(alpha = 1e-12) 
class InvariantsPurifier:
    def __init__(self, regressor = DEFAULT_LINEAR_REGRESSOR):
        self.regressor_ = clone(regressor)
        
    def fit(self, old_blocks, new_block):       
        old_uniting = unite_parallel(old_blocks)       
        self.regressor_.fit(old_uniting, new_block)
        
    def transform(self, old_blocks, new_block):
        old_uniting = unite_parallel(old_blocks)
        predictions = self.regressor_.predict(old_uniting)
        return subtract_parallel(new_block, predictions)
    
class CovariantsIndividualPurifier:
    def __init__(self, regressor = DEFAULT_LINEAR_REGRESSOR):
        self.regressor_ = clone(regressor)
        self.regressor_.set_params(**{"fit_intercept" : False})
        
    def fit(self, old_blocks, new_block, l):        
        old_blocks_reshaped = [pack_dense(old_block, l, old_block.shape[1], old_block.shape[1]) \
                               for old_block in old_blocks]
        old_uniting = unite_parallel(old_blocks_reshaped)
        new_reshaped = pack_dense(new_block, l, new_block.shape[1], new_block.shape[1])
        self.regressor_.fit(old_uniting, new_reshaped)
        
    def transform(self, old_blocks, new_block, l):
        old_blocks_reshaped = [pack_dense(old_block, l, old_block.shape[1], old_block.shape[1]) \
                               for old_block in old_blocks]
        old_uniting = unite_parallel(old_blocks_reshaped)
        new_reshaped = pack_dense(new_block, l, new_block.shape[1], new_block.shape[1])
        predictions = self.regressor_.predict(old_uniting)
        result = subtract_parallel(new_reshaped, predictions)        
        return unpack_dense(result, new_block.shape[0], l, new_block.shape[1])
        
class CovariantsPurifier:
    def __init__(self, regressor = DEFAULT_LINEAR_REGRESSOR):
        self.regressor_ = clone(regressor)
        self.regressor_.set_params(**{"fit_intercept" : False})
        
    def fit(self, old_datas, new_data):
        self.l_max_ = new_data.covariants_.shape[2] - 1
        self.purifiers_ = []
        
        for l in range(self.l_max_ + 1):
            self.purifiers_.append(CovariantsIndividualPurifier(regressor = clone(self.regressor_))) 
      
        for l in range(self.l_max_ + 1):            
            old_blocks_now = [old_data.covariants_[:, :old_data.actual_sizes_[l], l, :] \
                              for old_data in old_datas if (old_data.actual_sizes_[l] > 0) ]
            new_block_now = new_data.covariants_[:, :new_data.actual_sizes_[l], l, :]
            
            old_total_size = 0
            for old_data in old_datas:
                old_total_size += old_data.actual_sizes_[l]
            new_size = new_data.actual_sizes_[l]
            if (old_total_size == 0) or (new_size == 0):
                self.purifiers_[l] = None
            else:
                self.purifiers_[l].fit(old_blocks_now, new_block_now, l)
           
    def transform(self, old_datas, new_data):
        ans = Data(np.empty(new_data.covariants_.shape), np.copy(new_data.actual_sizes_),
                      importances = np.copy(new_data.importances_),
                      raw_importances = np.copy(new_data.raw_importances_))
        
        for l in range(self.l_max_ + 1):
            if (self.purifiers_[l] is not None):
                old_blocks_now = [old_data.covariants_[:, :old_data.actual_sizes_[l], l, :] for old_data in old_datas]
                new_block_now = new_data.covariants_[:, :new_data.actual_sizes_[l], l, :]
                now = self.purifiers_[l].transform(old_blocks_now, new_block_now, l)
                ans.covariants_[:, :now.shape[1], l, :(2 * l + 1)] = now # todo parallelize it
            else:                
                if (ans.actual_sizes_[l] > 0):                   
                    ans.covariants_[:, :ans.actual_sizes_[l], l, :(2 * l + 1)] = \
                    new_data.covariants_[:, :ans.actual_sizes_[l], l, :(2 * l + 1)] #todo parallelize it
                   
        return ans
    
class CovariantsPurifierBoth:
    def __init__(self, regressor = DEFAULT_LINEAR_REGRESSOR):
        self.even_purifier_ = CovariantsPurifier(regressor = clone(regressor))
        self.odd_purifier_ = CovariantsPurifier(regressor = clone(regressor))
        
    def fit(self, old_datas_even, new_data_even, old_datas_odd, new_data_odd):
        self.even_purifier_.fit(old_datas_even, new_data_even)
        self.odd_purifier_.fit(old_datas_odd, new_data_odd)
        
    def transform(self, old_datas_even, new_data_even, old_datas_odd, new_data_odd):
        return self.even_purifier_.transform(old_datas_even, new_data_even),\
               self.odd_purifier_.transform(old_datas_odd, new_data_odd)