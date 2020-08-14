from libc.math cimport sin, M_PI, sqrt
cimport cython
import numpy as np
#from unrolling_individual_pca import IndividualLambdaPCAs
from cython.parallel import prange, threadid
from multiprocessing import cpu_count
from libc.stdlib cimport malloc, free
import warnings

cdef double sqrt_2 = sqrt(2.0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int min_c(int a, int b) nogil:
    if (a < b):
        return a
    else:
        return b
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_c(int a, int b) nogil:
    if (a > b):
        return a
    else:
        return b
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int abs_c(int a) nogil:
    if (a >= 0):
        return a
    else:
        return -a


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void unpack(double* covariant, int l, double* real, double* imag) nogil:
    
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
cdef void single_contraction(const double[:, :, :, :, :] clebsh_gordan,
                            double* first_covariant, int l1,
                            double* second_covariant, int l2,
                            int lambd, double* ans_placeholder, 
                            double** buff) nogil:
    
    
    unpack(first_covariant, l1, buff[0], buff[1])
    unpack(second_covariant, l2, buff[2], buff[3])
    
    cdef int mu, m2, m1
    cdef double real_now, imag_now
   
    cdef double tmp
    for mu in range(0, lambd + 1):
        real_now = 0.0
        imag_now = 0.0
        
#        for m2 in range(-l2, l2 + 1):
#            m1 = mu - m2
#            if (m1 >=-l1) and (m1 <= l1):
        for m2 in range(max(-l2, mu-l1), min(l2,mu+l1)+1):
                m1 = mu - m2
                real_now += clebsh_gordan[l1, l2, lambd, m1 + l1, m2 + l2] * \
                (buff[0][l1 + m1] * buff[2][l2 + m2] - buff[1][l1 + m1] * buff[3][l2 + m2])
                
                imag_now += clebsh_gordan[l1, l2, lambd, m1 + l1, m2 + l2] * \
                    (buff[0][l1 + m1] * buff[3][l2 + m2] + buff[1][l1 + m1] * buff[2][l2 + m2])
                
       
            
        if ((l1 + l2 - lambd) % 2 == 1):
            tmp = real_now
            real_now = -imag_now
            imag_now = tmp
        
        if (mu > 0):
            ans_placeholder[mu + lambd] = real_now
            ans_placeholder[-mu + lambd] = imag_now
        else:
            ans_placeholder[lambd] = real_now / sqrt_2
            
  
'''def do_partial_expansion(clebsch_gordan, first_covariants, second_covariants, l_max,
                         tasks, res, res_actual_sizes, mode):
    if (mode == 'covariants'):
        do_partial_expansion_covariants(clebsch_gordan, first_covariants, second_covariants, l_max,
                                        tasks, res, res_actual_sizes)
    if (mode == 'invariants'):
        do_partial_expansion_invariants(clebsch_gordan, first_covariants, second_covariants, l_max,
                                        tasks, res, res_actual_sizes)'''


cdef void delete_buff(double*** buff, int num_threads):
    cdef int i, j
    for i in range(num_threads):
        for j in range(4):
            free(buff[i][j])
    for i in range(num_threads):
        free(buff[i])
    free(buff)
   

def process_contiguousness(array):
    if (array.flags['C_CONTIGUOUS']):
        return array
    else:
        warnings.warn("input and output covariants in partial expansion must be c contigous for proper cython parallelization. Making them contigous on single core. This can take a lot of time and it is not expectable behaviour. Try to provide contigous arrays for input")
        return np.ascontiguousarray(array)
    
def do_partial_expansion(clebsch_gordan, first_covariants, second_covariants, l_max,
                         tasks, res, res_actual_sizes, mode, num_threads = None):
    
    first_covariants = process_contiguousness(first_covariants)
    second_covariants = process_contiguousness(second_covariants)
    res = process_contiguousness(res)
    
    if (num_threads is None):
        num_threads = cpu_count()
    if (mode == "invariants"):
        do_partial_expansion_invariants(clebsch_gordan, first_covariants, second_covariants, l_max,
                                        tasks, res, res_actual_sizes, num_threads)
    if (mode == "covariants"):
        do_partial_expansion_covariants(clebsch_gordan, first_covariants, second_covariants, l_max,
                         tasks, res, res_actual_sizes, num_threads)
@cython.boundscheck(False)
@cython.wraparound(False)    
cpdef do_partial_expansion_covariants(const double[:, :, :, :, :] clebsch_gordan,
                        double[:, :, :, :] first_covariants,                       
                        double[:, :, :, :] second_covariants,                       
                        int l_max, int [:, :] tasks, 
                        double[:, :, :, :] res, int[:] res_actual_sizes, int num_threads):
    
    cdef int n_envs = first_covariants.shape[0]
    cdef int env_ind, first_ind, second_ind, lambd, l1, l2
    cdef int task_ind
    
    buff = get_buff(num_threads, l_max)
    cdef double** buff_now
    
    cdef int[:, :] placing = get_placing_covariants(l_max, tasks)
    
    for env_ind in prange(n_envs, nogil = True, schedule = "static", num_threads = num_threads):
        buff_now = buff[threadid()]
        for task_ind in range(tasks.shape[0]):            
            first_ind, l1 = tasks[task_ind, 0], tasks[task_ind, 1]
            second_ind, l2 = tasks[task_ind, 2], tasks[task_ind, 3]
            for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
               
                single_contraction(clebsch_gordan, &first_covariants[env_ind, first_ind, l1, 0], l1, 
                                   &second_covariants[env_ind, second_ind, l2, 0], l2,
                                   lambd, &res[env_ind, res_actual_sizes[lambd] + placing[task_ind, lambd], lambd, 0], buff_now)
                
    cdef int[:] sizes = get_sizes_covariants(l_max, tasks)
    for lambd in range(l_max + 1):
        res_actual_sizes[lambd] += sizes[lambd]
        
    delete_buff(buff, num_threads)
  
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef do_partial_expansion_invariants(const double[:, :, :, :, :] clebsch_gordan,
                        double[:, :, :, :] first_covariants,                       
                        double[:, :, :, :] second_covariants,                       
                        int l_max, int [:, :] tasks, 
                        double[:, :, :] res, int[:] res_actual_sizes, int num_threads): #last dim is dummy for m
    
    cdef int n_envs = first_covariants.shape[0]
    cdef int env_ind, first_ind, second_ind, l1, l2
    cdef int task_ind
    
    buff = get_buff(num_threads, l_max)
    cdef double** buff_now
    
    cdef int[:] placing = get_placing_invariants(l_max, tasks)
        
    for env_ind in prange(n_envs, nogil = True, schedule = "static", num_threads = num_threads):
        buff_now = buff[threadid()]
        for task_ind in range(tasks.shape[0]):            
            first_ind, l1 = tasks[task_ind, 0], tasks[task_ind, 1]
            second_ind, l2 = tasks[task_ind, 2], tasks[task_ind, 3]
            if (l1 == l2):
               
                single_contraction(clebsch_gordan, &first_covariants[env_ind, first_ind, l1, 0], l1, 
                                   &second_covariants[env_ind, second_ind, l2, 0], l2,
                                   0, &res[env_ind, res_actual_sizes[0] + placing[task_ind], 0], buff_now)
                
    res_actual_sizes[0] += get_size_invariants(l_max, tasks)
    delete_buff(buff, num_threads)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_size_invariants(int l_max, int[:, :] tasks):
    cdef int size = 0
    
    cdef int task_ind
    cdef int l1, l2
        
    for task_ind in range(tasks.shape[0]):
        l1, l2 = tasks[task_ind, 1], tasks[task_ind, 3]
        if (l1 == l2):
            size += 1       
    return size

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_sizes_covariants(int l_max, const int[:, :] tasks):
    sizes = np.zeros([l_max + 1], dtype = np.int32)
    cdef int[:] sizes_view = sizes
    cdef int task_ind
    cdef int lambd, l1, l2
        
    for task_ind in range(tasks.shape[0]):
        l1, l2 = tasks[task_ind, 1], tasks[task_ind, 3]
        for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
            sizes_view[lambd] += 1
    return sizes
    
def get_sizes(l_max, tasks, mode):
    if (mode == 'invariants'):
        return get_size_invariants(l_max, tasks)
    if (mode == 'covariants'):
        return get_sizes_covariants(l_max, tasks)
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:, :] get_placing_covariants(int l_max, const int[:, :] tasks):
    cdef int[:] now = np.zeros([l_max + 1], dtype = np.int32)
    result = np.zeros([tasks.shape[0], l_max + 1], dtype = np.int32)
    cdef int[:, :] result_view = result
    cdef int task_ind, lambd, l1, l2
    
    for task_ind in range(tasks.shape[0]):
        l1, l2 = tasks[task_ind, 1], tasks[task_ind, 3]
        for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
            result_view[task_ind, lambd] = now[lambd]
            now[lambd] += 1
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:] get_placing_invariants(int l_max, const int[:, :] tasks):
    cdef int now = 0
    result = np.zeros([tasks.shape[0]], dtype = np.int32)
    cdef int[:] result_view = result
    cdef int task_ind, l1, l2
    
    for task_ind in range(tasks.shape[0]):
        l1, l2 = tasks[task_ind, 1], tasks[task_ind, 3]
        if (l1 == l2):
            result_view[task_ind] = now
            now += 1
    return result

def get_placing(l_max, tasks, mode):
    if (mode == 'invariants'):
        return get_placing_invariants(l_max, tasks)
    if (mode == 'covariants'):
        return get_placing_covariants(l_max, tasks)
    

cdef double*** get_buff(int num_threads, int l_max):
    cdef double*** buff = <double ***> malloc(sizeof(double**) * num_threads)
    cdef int i, j
    for i in range(num_threads):
        buff[i] = <double **> malloc(sizeof(double*) * 4)
    for i in range(num_threads):
        for j in range(4):
            buff[i][j] = <double *> malloc(sizeof(double) * (2 * l_max + 1))
    return buff
    
    

'''cpdef do_full_expansion(double[:, :, :, :, :] clebsch_gordan,
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
                        res_actual_sizes[lambd] += 1'''
                               
                                
                                
                                
                
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
    def __init__(self, covariants, actual_sizes, importances = None):
        self.covariants_ = covariants
        self.actual_sizes_ = actual_sizes
        self.importances_ = importances       
        
    def get_invariants(self):
        return self.covariants_[:, :self.actual_sizes_[0], 0, 0]
    
    def __getitem__(self, sliced): #only env dim slice should be used
        return Data(self.covariants_[sliced], self.actual_sizes_[sliced], self.importances_)
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
