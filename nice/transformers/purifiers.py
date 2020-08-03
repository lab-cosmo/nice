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
from sklearn.exceptions import NotFittedError 

DEFAULT_LINEAR_REGRESSOR = Ridge(alpha = 1e-12) 
class InvariantsPurifier:
    def __init__(self, regressor = DEFAULT_LINEAR_REGRESSOR, num_to_fit = '10x', max_take = None):
        self.regressor_ = clone(regressor)
        self.fitted_ = False
        self.num_to_fit_ = num_to_fit
        self.max_take_ = max_take
        if (type(self.max_take_) == list):
            self.max_take_ = np.array(self.max_take_)
        if (self.max_take_ is not None) and (type(self.max_take_) != np.ndarray):
            self.max_take_ = int(self.max_take_)
        
        
    def fit(self, old_blocks, new_block):
        total_num = 0
        for i in range(len(old_blocks)):
            if (self.max_take_ is None):
                total_num += old_blocks[i].shape[1]
            else:
                if (type(self.max_take_) is int):
                    total_num += min(old_blocks[i].shape[1], self.max_take_)
                else:
                    total_num += min(old_blocks[i].shape[1], self.max_take_[i])
        
        if (type(self.num_to_fit_) is str):
            multiplier = int(parse('{}x', self.num_to_fit_)[0])                     
            num_fit_now = multiplier * total_num
        else:
            num_fit_now = self.num_to_fit_
            
        if (num_fit_now > new_block.shape[0]):
            warnings.warn("Amount of provided data is less than the desired one to fit InvariantsPurifer. Number of old features is {}, desired number of environments is {}, actual number of environments is {}".format(total_num, num_fit_now, new_block.shape[0]))
            
        if (self.max_take_ is None):
            restricted_blocks = [old_block[:num_fit_now, :] for old_block in old_blocks]
        else:
            if (type(self.max_take_) is int):
                restricted_blocks = [old_block[:num_fit_now, :self.max_take_] for old_block in old_blocks]
            else:
                restricted_blocks = [old_blocks[i][:num_fit_now, :self.max_take_[i]] for i in range(len(old_blocks))]       
        
        
        old_uniting = unite_parallel(restricted_blocks)       
        self.regressor_.fit(old_uniting, new_block[:num_fit_now, :])        
                    
        self.fitted_ = True
        
    def transform(self, old_blocks, new_block):
        if (not self.fitted_):
            raise NotFittedError("instance of {} is not fitted. It can not transform anything".format(type(self).__name__))
            
        if (self.max_take_ is None):
            restricted_blocks = [old_block[:, :] for old_block in old_blocks]
        else:
            if (type(self.max_take_) is int):
                restricted_blocks = [old_block[:, :self.max_take_] for old_block in old_blocks]
            else:
                restricted_blocks = [old_blocks[i][:, :self.max_take_[i]] for i in range(len(old_blocks))]   
            
        old_uniting = unite_parallel(restricted_blocks)
        predictions = self.regressor_.predict(old_uniting)
        return subtract_parallel(new_block, predictions)
    
    def is_fitted(self):
        return self.fitted_
    
    
class CovariantsIndividualPurifier:
    def __init__(self, regressor = DEFAULT_LINEAR_REGRESSOR, num_to_fit = '10x', max_take = None):
        self.regressor_ = clone(regressor)
        self.regressor_.set_params(**{"fit_intercept" : False})
        self.fitted_ = False
        self.num_to_fit_ = num_to_fit
        self.max_take_ = max_take
        if (type(self.max_take_) == list):
            self.max_take_ = np.array(self.max_take_)
            
        if (self.max_take_ is not None) and (type(self.max_take_) != np.ndarray):
            self.max_take_ = int(self.max_take_)
            
    def fit(self, old_blocks, new_block, l):
        total_num = 0
        for i in range(len(old_blocks)):
            if (self.max_take_ is None):
                total_num += old_blocks[i].shape[1]
            else:
                if (type(self.max_take_) is int):
                    total_num += min(old_blocks[i].shape[1], self.max_take_)
                else:
                    total_num += min(old_blocks[i].shape[1], self.max_take_[i])

        if (type(self.num_to_fit_) is str):
            multiplier = int(parse('{}x', self.num_to_fit_)[0])                     
            num_fit_now = multiplier * total_num
        else:
            num_fit_now = self.num_to_fit_
            
        if (num_fit_now > new_block.shape[0] * (l + 1)):
            warnings.warn("Amount of provided data is less than the desired one to fit InvariantsPurifer. Number of old features is {}, desired number of data points is {}, actual number of data points (n_env * (l + 1)) is {}, since number of environments is {}, and l is {}".format(total_num, num_fit_now, new_block.shape[0] * (l + 1), new_block.shape[0], l)) 
        
        if (num_fit_now % (l + 1) == 0):
            num_fit_now = num_fit_now // (l + 1)
        else:
            num_fit_now = (num_fit_now // (l + 1)) + 1
            
        if (self.max_take_ is None):
             old_blocks_reshaped = [pack_dense(old_block[:num_fit_now], l, old_block.shape[1], old_block.shape[1]) \
                               for old_block in old_blocks]
        else:
            if (type(self.max_take_) is int):
                 old_blocks_reshaped = [pack_dense(old_block[:num_fit_now, :self.max_take_], l, self.max_take_, self.max_take_) \
                                   for old_block in old_blocks]
            else:
                old_blocks_reshaped = [pack_dense(old_blocks[i][:num_fit_now, :self.max_take_[i]], l, self.max_take_[i], self.max_take_[i])  for i in range(len(old_blocks))] 
            
            
       
        old_uniting = unite_parallel(old_blocks_reshaped)
        new_reshaped = pack_dense(new_block[:num_fit_now], l, new_block.shape[1], new_block.shape[1])
        self.regressor_.fit(old_uniting, new_reshaped)
        self.fitted_ = True
        
    def transform(self, old_blocks, new_block, l):
        if (not self.fitted_):
            raise NotFittedError("instance of {} is not fitted. It can not transform anything".format(type(self).__name__))
            
        if (self.max_take_ is None):
             old_blocks_reshaped = [pack_dense(old_block, l, old_block.shape[1], old_block.shape[1]) \
                               for old_block in old_blocks]
        else:
            if (type(self.max_take_) is int):
                 old_blocks_reshaped = [pack_dense(old_block[:, :self.max_take_], l, self.max_take_, self.max_take_) \
                                   for old_block in old_blocks]
            else:
                old_blocks_reshaped = [pack_dense(old_blocks[i][:, :self.max_take_[i]], l, self.max_take_[i], self.max_take_[i]) for i in range(len(old_blocks))]
            
        old_uniting = unite_parallel(old_blocks_reshaped)
        new_reshaped = pack_dense(new_block, l, new_block.shape[1], new_block.shape[1])
        predictions = self.regressor_.predict(old_uniting)
        result = subtract_parallel(new_reshaped, predictions)        
        return unpack_dense(result, new_block.shape[0], l, new_block.shape[1])
    
    def is_fitted(self):
        return self.fitted_
        
class CovariantsPurifier:
    def __init__(self, regressor = DEFAULT_LINEAR_REGRESSOR, num_to_fit = '10x', max_take = None):
        self.regressor_ = clone(regressor)
        self.regressor_.set_params(**{"fit_intercept" : False})
        self.fitted_ = False
        self.num_to_fit_ = num_to_fit
        self.max_take_ = max_take
        if (type(self.max_take_) == list):
            self.max_take_ = np.array(self.max_take_)
        if (self.max_take_ is not None) and (type(self.max_take_) != np.ndarray):
            self.max_take_ = int(self.max_take_)
            
    def fit(self, old_datas, new_data):
        
        self.l_max_ = new_data.covariants_.shape[2] - 1
        self.purifiers_ = []
        
        for l in range(self.l_max_ + 1):
            self.purifiers_.append(CovariantsIndividualPurifier(regressor = clone(self.regressor_), num_to_fit = self.num_to_fit_, max_take = self.max_take_)) 
      
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
                
        self.fitted_ = True
           
    def transform(self, old_datas, new_data):
        if (not self.fitted_):
            raise NotFittedError("instance of {} is not fitted. It can not transform anything".format(type(self).__name__))
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
    
    def is_fitted(self):
        return self.fitted_
    
class CovariantsPurifierBoth:
    def __init__(self, regressor = DEFAULT_LINEAR_REGRESSOR, num_to_fit = '10x', max_take = None):
        self.num_to_fit_ = num_to_fit
        self.max_take_ = max_take
        if (self.max_take_ is not None) and (type(self.max_take_) != np.ndarray):
            self.max_take_ = int(self.max_take_)
        self.even_purifier_ = CovariantsPurifier(regressor = clone(regressor), num_to_fit = self.num_to_fit_, max_take = self.max_take_)
        self.odd_purifier_ = CovariantsPurifier(regressor = clone(regressor), num_to_fit = self.num_to_fit_, max_take = self.max_take_)
        self.fitted_ = False
        
        
    def fit(self, old_datas_even, new_data_even, old_datas_odd, new_data_odd):
        
        self.even_purifier_.fit(old_datas_even, new_data_even)
        self.odd_purifier_.fit(old_datas_odd, new_data_odd)
        self.fitted_ = True
        
    def transform(self, old_datas_even, new_data_even, old_datas_odd, new_data_odd):
        if (not self.fitted_):
            raise NotFittedError("instance of {} is not fitted. It can not transform anything".format(type(self).__name__))
        return self.even_purifier_.transform(old_datas_even, new_data_even),\
               self.odd_purifier_.transform(old_datas_odd, new_data_odd)
    
    def is_fitted(self):
        return self.fitted_