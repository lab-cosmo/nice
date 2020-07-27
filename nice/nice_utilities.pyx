from libc.math cimport sin, M_PI, sqrt
cimport cython
import numpy as np
#from unrolling_individual_pca import IndividualLambdaPCAs
#from cython.parallel cimport prange

cdef double sqrt_2 = sqrt(2.0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int min_c(int a, int b):
    if (a < b):
        return a
    else:
        return b
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_c(int a, int b):
    if (a > b):
        return a
    else:
        return b
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int abs_c(int a):
    if (a >= 0):
        return a
    else:
        return -a


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void unpack(double[:] covariant, int l, double[:] real, double[:] imag):
    
    real[l] = sqrt_2 * covariant[l]
    cdef int m
    for m in range(1, l + 1):
        real[m + l] = covariant[m + l]
        if (m % 2 == 0):
            real[-m + l] = covariant[m + l]
        else:
            real[-m + l] = -covariant[m + l]
    
    imag[l] = 0.0
    for m in range(1, l + 1):
        imag[m + l] = covariant[-m + l]
        if (m % 2 == 0):
            imag[-m + l] = -covariant[-m + l]
        else:
            imag[-m + l] = covariant[-m + l]
            
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void single_contraction(double[:, :, :, :, :] clebsh_gordan,
                            double[:] first_covariant, int l1,
                            double[:] second_covariant, int l2,
                            int lambd, double[:] ans_placeholder, 
                            double[:, :] buff):
    
    
    unpack(first_covariant, l1, buff[0], buff[1])
    unpack(second_covariant, l2, buff[2], buff[3])
    
    cdef int mu, m2, m1
    cdef double real_now, imag_now
   
    cdef tmp
    for mu in range(0, lambd + 1):
        real_now = 0.0
        imag_now = 0.0
        
        for m2 in range(-l2, l2 + 1):
            m1 = mu - m2
            if (m1 >=-l1) and (m1 <= l1):
                real_now += clebsh_gordan[l1, l2, lambd, m1 + l1, m2 + l2] * \
                (buff[0, l1 + m1] * buff[2, l2 + m2] - buff[1, l1 + m1] * buff[3, l2 + m2])
                
                imag_now += clebsh_gordan[l1, l2, lambd, m1 + l1, m2 + l2] * \
                    (buff[0, l1 + m1] * buff[3, l2 + m2] + buff[1, l1 + m1] * buff[2, l2 + m2])
                
       
            
        if ((l1 + l2 - lambd) % 2 == 1):
            tmp = real_now
            real_now = -imag_now
            imag_now = tmp
        
        if (mu > 0):
            ans_placeholder[mu + lambd] = real_now
            ans_placeholder[-mu + lambd] = imag_now
        else:
            ans_placeholder[lambd] = real_now / sqrt_2
            
  
def do_partial_expansion(clebsch_gordan, first_covariants, second_covariants, l_max,
                         tasks, res, res_actual_sizes, mode):
    if (mode == 'covariants'):
        do_partial_expansion_covariants(clebsch_gordan, first_covariants, second_covariants, l_max,
                                        tasks, res, res_actual_sizes)
    if (mode == 'invariants'):
        do_partial_expansion_invariants(clebsch_gordan, first_covariants, second_covariants, l_max,
                                        tasks, res, res_actual_sizes)

cpdef do_partial_expansion_invariants(double[:, :, :, :, :] clebsch_gordan,
                        double[:, :, :, :] first_covariants,                       
                        double[:, :, :, :] second_covariants,                       
                        int l_max, int [:, :] tasks, 
                        double[:, :, :] res, int[:] res_actual_sizes): #last dim is dummy for m
    
    cdef int n_envs = first_covariants.shape[0]
    cdef int env_ind, first_ind, second_ind, l1, l2
    cdef int task_ind
    buff = np.empty([4, 2 * l_max + 1])
        
    for task_ind in range(tasks.shape[0]):
        first_ind, l1 = tasks[task_ind, 0], tasks[task_ind, 1]
        second_ind, l2 = tasks[task_ind, 2], tasks[task_ind, 3]
        if (l1 == l2):       
            for env_ind in range(n_envs):
                single_contraction(clebsch_gordan, first_covariants[env_ind, first_ind, l1], l1, 
                                                       second_covariants[env_ind, second_ind, l2], l2,
                                                       0, res[env_ind, res_actual_sizes[0]], buff)
            res_actual_sizes[0] += 1


cpdef get_sizes(int l_max, int[:, :] tasks):
    sizes = np.zeros([l_max + 1], dtype = np.int32)
    int[:] sizes_view = sizes
    cdef int task_ind
    cdef int lambd, l1, l2
        
    for task_ind in range(tasks.shape[0]):       
        for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
            sizes_view[lambd] += 1
    return sizes
    
cpdef do_partial_expansion_covariants(double[:, :, :, :, :] clebsch_gordan,
                        double[:, :, :, :] first_covariants,                       
                        double[:, :, :, :] second_covariants,                       
                        int l_max, int [:, :] tasks, 
                        double[:, :, :, :] res, int[:] res_actual_sizes):
    
    cdef int n_envs = first_covariants.shape[0]
    cdef int env_ind, first_ind, second_ind, lambd, l1, l2
    cdef int task_ind
    buff = np.empty([4, 2 * l_max + 1])
        
    for task_ind in range(tasks.shape[0]):
        first_ind, l1 = tasks[task_ind, 0], tasks[task_ind, 1]
        second_ind, l2 = tasks[task_ind, 2], tasks[task_ind, 3]
        for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
            for env_ind in range(n_envs):
                single_contraction(clebsch_gordan, first_covariants[env_ind, first_ind, l1], l1, 
                                                       second_covariants[env_ind, second_ind, l2], l2,
                                                       lambd,res[env_ind, res_actual_sizes[lambd], lambd], buff)
            res_actual_sizes[lambd] += 1

    
cpdef do_full_expansion(double[:, :, :, :, :] clebsch_gordan,
                        double[:, :, :, :] first_covariants,
                        int[:] first_actual_sizes,
                        double[:, :, :, :] second_covariants,
                        int[:] second_actual_sizes,
                        int l_max, double[:, :, :, :] res,int[:] res_actual_sizes):

    cdef int n_envs = first_covariants.shape[0]
    cdef int env_ind, first_num_now, second_num_now, first_ind, second_ind, lambd, l1, l2
    buff = np.empty([4, 2 * l_max + 1])
    
    
    for l1 in range(l_max + 1):
        first_num_now = first_actual_sizes[l1]
        for l2 in range(l_max + 1):
            second_num_now = second_actual_sizes[l2]
            for first_ind in range(first_num_now):
                for second_ind in range(second_num_now):
                    for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
                        for env_ind in range(n_envs):
                            single_contraction(clebsch_gordan, first_covariants[env_ind, first_ind, l1], l1, 
                                                   second_covariants[env_ind, second_ind, l2], l2,
                                                   lambd,res[env_ind, res_actual_sizes[lambd], lambd], buff)
                        res_actual_sizes[lambd] += 1
                               
                                
                                
                                
                
'''cpdef do_full_expansion(double[:, :, :, :, :] clebsh_gordan,
                   double[:, :, :, :] current_features, 
                   double[:, :, :, :] expansion_coefficients,
                   int l_max):
    
    cdef int n_envs = current_features.shape[0]
    cdef int n_radial = expansion_coefficients.shape[1]
    cdef int n_feat_now = current_features.shape[1]
    cdef int env_ind, current_ind, coef_ind, l1, l2, lambd, pos_now
    
    cdef int n_same = (l_max + 2) // 2
    cdef int n_other = (l_max + 1) // 2
    cdef int multiplier = n_feat_now * n_radial * (l_max + 1)
    
    new_same_parity_features = np.zeros([n_envs, n_same * multiplier, l_max + 1, 2 * l_max + 1])
    new_other_parity_features = np.zeros([n_envs, n_other * multiplier, l_max + 1, 2 * l_max + 1])
    buff = np.empty([4, 2 * l_max + 1])
    
    cdef double[:, :, :, :] same_view = new_same_parity_features
    cdef double[:, :, :, :] other_view = new_other_parity_features
    
    for env_ind in range(n_envs):
        pos_same_now = 0
        pos_other_now = 0
        for current_ind in range(n_feat_now):
            for coef_ind in range(n_radial):
                for l1 in range(l_max + 1):
                    for l2 in range(l_max + 1):
                        if (l2 % 2 == 0):
                            for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
                                single_contraction(clebsh_gordan, current_features[env_ind, current_ind, l1], l1, 
                                                   expansion_coefficients[env_ind, coef_ind, l2], l2,
                                                   lambd, same_view[env_ind, pos_same_now, lambd], buff)
                            pos_same_now += 1
                        else:
                            for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
                                single_contraction(clebsh_gordan, current_features[env_ind, current_ind, l1], l1, 
                                                   expansion_coefficients[env_ind, coef_ind, l2], l2,
                                                   lambd, other_view[env_ind, pos_other_now, lambd], buff)
                           
                            pos_other_now += 1
                            
    return new_same_parity_features, new_other_parity_features'''

class Data:
    def __init__(self, covariants, actual_sizes, importances = None, raw_importances = None):
        self.covariants_ = covariants
        self.actual_sizes_ = actual_sizes
        self.importances_ = importances
        self.raw_importances_ = raw_importances
        
    def get_invariants(self):
        return self.covariants_[:, :self.actual_sizes_[0], 0, 0]
    
    def __getitem__(self, sliced): #only env dim slice should be used
        return Data(self.covariants_[sliced], self.actual_sizes_[sliced], self.importances_, self.raw_importances_)
        '''args = [self.covariants_[sliced], self.actual_sizes_[sliced]]
        if (self.importances_ is None):
            args.append(None)
        else:
            args.append(self.importances_)
        if (self.importances_ is None):            
            return Data(self.covariants_[sliced], self.actual_sizes_[sliced])
        else:
            return Data(self.covariants_[sliced], self.actual_sizes_[sliced], self.importances_)'''
    
'''def do_initial_transform(coefficients, l_max):   

    even_coefficients = np.copy(coefficients)
    even_coefficients_sizes = [coefficients.shape[1] if i % 2 == 0 else 0 for i in range(l_max + 1)]

    odd_coefficients = np.copy(coefficients)
    odd_coefficients_sizes = [coefficients.shape[1] if i % 2 == 1 else 0 for i in range(l_max + 1)]

    even_pcas = IndividualLambdaPCAs(n_components = coefficients.shape[1])
    even_pcas.fit(even_coefficients, even_coefficients_sizes)

    odd_pcas = IndividualLambdaPCAs(n_components = coefficients.shape[1])
    odd_pcas.fit(odd_coefficients, odd_coefficients_sizes)

    even, even_sizes = even_pcas.transform(even_coefficients, even_coefficients_sizes)
    even_importances = even_pcas.get_importances()

    odd, odd_sizes = odd_pcas.transform(odd_coefficients, odd_coefficients_sizes)
    odd_importances = odd_pcas.get_importances()

    return even, even_sizes, even_importances, odd, odd_sizes, odd_importances'''