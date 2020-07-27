import numpy as np
from nice.unrolling_individual_pca import UnrollingIndividualPCA
#from cython.parallel cimport prange

from nice.thresholding import get_thresholded_tasks
from nice.nice_utilities import do_partial_expansion, Data, get_sizes
from nice.ClebschGordan import ClebschGordan
from parse import parse
import warnings

class ThresholdExpansioner:
    def __init__(self, num_expand = None, mode = 'covariants'):
        if (num_expand is None):
            self.num_expand_ = -1
        else:
            self.num_expand_ = num_expand
        
        self.mode_ = mode
        
    
    def fit(self, first_even, first_odd, second_even, second_odd):        
        self.l_max_ = first_even.covariants_.shape[2] - 1
        self.task_even_even_, self.task_odd_odd_, self.task_even_odd_, self.task_odd_even_ = \
        get_thresholded_tasks(first_even, first_odd, second_even, second_odd, self.num_expand_, self.l_max_, self.mode_) 
        
        self.clebsch_ = ClebschGordan(self.l_max_)
        self.new_even_size_ = np.max(get_sizes(self.l_max_, self.task_even_even_[0], self.mode_) + \
                        get_sizes(self.l_max_, self.task_odd_odd_[0], self.mode_))
        
        self.new_odd_size_ = np.max(get_sizes(self.l_max_, self.task_even_odd_[0], self.mode_) + \
                        get_sizes(self.l_max_, self.task_odd_even_[0], self.mode_))
        
    def transform(self, first_even, first_odd, second_even, second_odd):
       
        
        if (self.mode_ == 'covariants'):
            new_even = np.empty([first_even.covariants_.shape[0], self.new_even_size_, self.l_max_ + 1, 2 * self.l_max_ + 1])
            new_odd = np.empty([first_even.covariants_.shape[0], self.new_odd_size_, self.l_max_ + 1, 2 * self.l_max_ + 1])
        else:
            new_even = np.empty([first_even.covariants_.shape[0], self.new_even_size_, 1])
            new_odd = np.empty([first_even.covariants_.shape[0], self.new_odd_size_, 1])
        
        if (self.mode_ == 'covariants'):
            new_even_actual_sizes = np.zeros([self.l_max_ + 1], dtype = np.int32)
            new_odd_actual_sizes = np.zeros([self.l_max_ + 1], dtype = np.int32)
        else:
            new_even_actual_sizes = np.zeros([1], dtype = np.int32)
            new_odd_actual_sizes = np.zeros([1], dtype = np.int32)
        
       
        do_partial_expansion(self.clebsch_.precomputed_, first_even.covariants_,
                             second_even.covariants_,
                             self.l_max_, self.task_even_even_[0], new_even, new_even_actual_sizes, self.mode_)
        #print(new_even_actual_sizes)
        do_partial_expansion(self.clebsch_.precomputed_, first_odd.covariants_,
                             second_odd.covariants_,
                             self.l_max_, self.task_odd_odd_[0], new_even, new_even_actual_sizes, self.mode_)
        #print(new_even_actual_sizes)
        do_partial_expansion(self.clebsch_.precomputed_, first_even.covariants_,
                             second_odd.covariants_,
                             self.l_max_, self.task_even_odd_[0], new_odd, new_odd_actual_sizes, self.mode_)
        
        do_partial_expansion(self.clebsch_.precomputed_, first_odd.covariants_,
                             second_even.covariants_,
                             self.l_max_, self.task_odd_even_[0], new_odd, new_odd_actual_sizes, self.mode_) 
        if (self.mode_ == 'covariants'):
            return Data(new_even, new_even_actual_sizes), Data(new_odd, new_odd_actual_sizes)
        else:
            return new_even[:, :new_even_actual_sizes[0], 0], new_odd[:, :new_odd_actual_sizes[0], 0]
        

def get_num_fit(desired_num, block_size):
    if (desired_num % block_size == 0):
        return desired_num // block_size
    else:
        return (desired_num // block_size) + 1

class IndividualLambdaPCAs():    
    def __init__(self, n_components = None, num_to_fit = '10x'):
        self.n_components_ = n_components
        self.num_to_fit_ = num_to_fit
        
    def get_importances(self):        
        result = np.empty([self.max_n_components_, self.l_max_ + 1])
        for lambd in range(self.l_max_ + 1):
            if (self.pcas_[lambd] is not None):
                result[:self.pcas_[lambd].n_components, lambd] = self.pcas_[lambd].importances_
            
        actual_sizes = []
        for lambd in range(self.l_max_ + 1):
            if (self.pcas_[lambd] is not None):
                actual_sizes.append(self.pcas_[lambd].n_components)
            else:
                actual_sizes.append(0)        
        return result 
    
    def fit(self, data):
        self.l_max_ = data.covariants_.shape[2] - 1
        self.pcas_ = []
        self.reduction_happened_ = False
        self.max_n_components_ = -1 
        for lambd in range(self.l_max_ + 1):
            if (data.actual_sizes_[lambd] > 0):
                if (self.n_components_ is None):
                    n_components_now = data.actual_sizes_[lambd]
                else:
                    n_components_now = self.n_components_
                    
                self.max_n_components_ = max(self.max_n_components_, n_components_now)
                
                if (data.covariants_.shape[0] * (lambd + 1) < n_components_now):
                    raise ValueError("not enough data to fit pca, number of vectors is {}, dimensionality of single vector (lambd + 1) is {}, i. e. total number of points is {}, while number of components is {}".format(data.covariants_.shape[0], 
                    lambd + 1, data.covariants_.shape[0] * (lambd + 1), n_components_now))
                
                if (type(self.num_to_fit_) is str):
                    multiplier = int(parse('{}x', self.num_to_fit_)[0])                     
                    num_fit_now = get_num_fit(multiplier * n_components_now, (lambd + 1))
                else:
                    num_fit_now = self.num_to_fit_
                    if (num_fit_now * (lambd + 1) < n_components_now):
                        raise ValueError("specified parameter num fit ({}) is too small to fit pca with number of components {} ".format(num_fit_now, n_components_now))
                          
                      
                if (data.covariants_.shape[0] * (lambd + 1) < num_fit_now):
                    warnings.warn("given data is less than desired number of points to fit pca. Desired number of points to fit pca is {}, while number of vectors is {}, dimensionality of single vector (lambd + 1) is {}, i. e. total number of points is {}. Number of pca components is {}".format(num_fit_now, data.covariants_.shape[0], (lambd + 1), data.covariants_.shape[0] * (lambd + 1), n_components_now), RuntimeWarning)

                                     
                if (n_components_now < data.actual_sizes_[lambd]):
                    self.reduction_happened_ = True
                pca = UnrollingIndividualPCA(n_components = n_components_now)
                pca.fit(data.covariants_[:num_fit_now, :, lambd, :], data.actual_sizes_[lambd], lambd)
                self.pcas_.append(pca)
            else:
                self.pcas_.append(None)
                
        self.importances_ = self.get_importances()
            
    def transform(self, data):
        result = np.empty([data.covariants_.shape[0], self.max_n_components_, self.l_max_ + 1, 2 * self.l_max_ + 1])
        new_actual_sizes = np.zeros([self.l_max_ + 1], dtype = np.int32)
        for lambd in range(self.l_max_ + 1):
            if (self.pcas_[lambd] is not None):
                now = self.pcas_[lambd].transform(data.covariants_[:, :, lambd, :], data.actual_sizes_[lambd], lambd)
                result[:, :now.shape[1], lambd, :(2*lambd + 1)] = now
                new_actual_sizes[lambd] = now.shape[1]
            else:
                new_actual_sizes[lambd] = 0
            
        return Data(result, new_actual_sizes, importances = self.importances_)    
    
    
class IndividualLambdaPCAsBoth():
    def __init__(self, *args, **kwargs):
        self.even_pca_ = IndividualLambdaPCAs(*args, **kwargs)
        self.odd_pca_ = IndividualLambdaPCAs(*args, **kwargs)
        
    def fit(self, data_even, data_odd):
        self.even_pca_.fit(data_even)
        self.odd_pca_.fit(data_odd)
    
    def transform(self, data_even, data_odd):
        return self.even_pca_.transform(data_even), self.odd_pca_.transform(data_odd)
    
class InitialTransformer():
    def transform(self, coefficients):
        l_max = coefficients.shape[2] - 1
        even_coefficients = np.copy(coefficients)
        even_coefficients_sizes = [coefficients.shape[1] if i % 2 == 0 else 0 for i in range(l_max + 1)]

        odd_coefficients = np.copy(coefficients)
        odd_coefficients_sizes = [coefficients.shape[1] if i % 2 == 1 else 0 for i in range(l_max + 1)]
        
        return Data(even_coefficients, even_coefficients_sizes), Data(odd_coefficients, odd_coefficients_sizes)
    
class StandardBlock():
    def __init__(self, covariants_expansioner = None, covariants_pca = None, invariants_expansioner = None,
                 invariants_pca = None):
        
        self.covariants_expansioner_ = covariants_expansioner
        if (self.covariants_expansioner_ is not None):
            if self.covariants_expansioner_.mode_ != 'covariants':
                raise ValueError("mode of covariants expansioner should be covariants")  
        
        self.covariants_pca_ = covariants_pca
        if (self.covariants_pca_ is not None) and (self.covariants_expansioner_ is None):
            raise ValueError("can not do pca over not existing covariants")
        
        self.invariants_expansioner_ = invariants_expansioner
        if (self.invariants_expansioner_ is not None):
            if self.invariants_expansioner_.mode_ != 'invariants':
                raise ValueError("mode of invariants expansioner should be invariants")
        
        self.invariants_pca_ = invariants_pca
        if (self.invariants_pca_ is not None) and (self.invariants_expansioner_ is None):
            raise ValueError("can not do pca over not existing invariants")
            
        if (self.covariants_expansioner_ is not None) and (self.covariants_pca_ is not None):
            self.higher_body_orders_possible_ = True
        else:
            self.higher_body_orders_possible_ = False
            
        if (self.covariants_expansioner_ is None) and (self.invariants_expansioner_ is None):
            raise ValueError("nothing to do")
        
    def fit(self, first_even, first_odd, second_even, second_odd):
        if (self.covariants_expansioner_ is not None):
            self.covariants_expansioner_.fit(first_even, first_odd, second_even, second_odd)
        if (self.covariants_pca_ is not None):
            transformed_even, transformed_odd = self.covariants_expansioner_.transform(first_even, first_odd, second_even, second_odd)
           
            self.covariants_pca_.fit(transformed_even, transformed_odd)
            
        if (self.invariants_expansioner_ is not None):
            self.invariants_expansioner_.fit(first_even, first_odd, second_even, second_odd)
        if (self.invariants_pca_ is not None):
            invariants_even, _ = self.invariants_expansioner_.transform(first_even, first_odd, second_even, second_odd)
            self.invariants_pca_.fit(invariants_even)
        
    def transform(self, first_even, first_odd, second_even, second_odd):
        transformed_even, transformed_odd = None, None
        if (self.covariants_expansioner_ is not None):
            transformed_even, transformed_odd = self.covariants_expansioner_.transform(first_even, first_odd, second_even, second_odd)
        if (self.covariants_pca_ is not None):
            transformed_even, transformed_odd = self.covariants_pca_.transform(transformed_even, transformed_odd)
        
        invariants_even = None
        if (self.invariants_expansioner_ is not None):
            invariants_even, _ = self.invariants_expansioner_.transform(first_even, first_odd, second_even, second_odd)
        if (self.invariants_pca_ is not None):
            invariants_even = self.invariants_pca_.transform(invariants_even)
       
        return transformed_even, transformed_odd, invariants_even
        
    
class StandardSequence():
    def __init__(self, blocks, initial_pca = IndividualLambdaPCAsBoth()):
        self.blocks_ = blocks
        self.initial_pca_ = initial_pca
        for i in range(len(self.blocks_) - 1):
            if not self.blocks_[i].higher_body_orders_possible_:
                raise ValueError("all intermediate standard blocks should calculate covariants")
        self.initial_transformer_ = InitialTransformer()
                
    def fit(self, coefficients):
        self.intermediate_sizes_ = []
        data_even_0, data_odd_0 = self.initial_transformer_.transform(coefficients)
        self.initial_pca_.fit(data_even_0, data_odd_0)
        data_even_0, data_odd_0 = self.initial_pca_.transform(data_even_0, data_odd_0)
        data_even_now, data_odd_now = data_even_0, data_odd_0
        self.intermediate_sizes_.append([data_even_now.actual_sizes_, data_odd_now.actual_sizes_])
        for i in range(len(self.blocks_)):
            self.blocks_[i].fit(data_even_now, data_odd_now, data_even_0, data_odd_0)            
            data_even_now, data_odd_now, _ = self.blocks_[i].transform(data_even_now, data_odd_now, data_even_0, data_odd_0)
            if (data_even_now is not None):
                self.intermediate_sizes_.append([data_even_now.actual_sizes_, data_odd_now.actual_sizes_])
        
    def transform(self, coefficients, return_only_invariants = False):
        data_even_0, data_odd_0 = self.initial_transformer_.transform(coefficients)
        data_even_0, data_odd_0 = self.initial_pca_.transform(data_even_0, data_odd_0)
        
        
        all_invariants = [data_even_0.get_invariants()]
        data_even_now, data_odd_now = data_even_0, data_odd_0
        for i in range(len(self.blocks_)):
            data_even_now, data_odd_now, invariants_even_now = self.blocks_[i].transform(data_even_now, data_odd_now, data_even_0, data_odd_0)
            
            if (invariants_even_now is not None):
                all_invariants.append(invariants_even_now)
            else:
                all_invariants.append(data_even_now.get_invariants())
        
        if (return_only_invariants):
            return all_invariants
        else:
            return data_even_now, data_odd_now, all_invariants
        
            
    
