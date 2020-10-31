from libc.math cimport sin, M_PI, sqrt, fmax
cimport cython
import numpy as np
#from cython.parallel cimport prange
from nice_utilities cimport min_c, abs_c, max_c
from nice_utilities import Data
cdef double sqrt_2 = sqrt(2.0)
#from nice_utilities import Data

cdef enum Mode:
    covariants, invariants

cdef enum Question:
    n_pairs, n_tasks
  
cdef get_thresholded_task(double[:, :] first_importances, int[:] first_actual_sizes,
                           double[:, :] second_importances, int[:] second_actual_sizes,
                           double threshold, int known_num, int l_max, Mode mode):
    if mode == Mode.covariants:
        return get_thresholded_task_covariants(first_importances, first_actual_sizes,
                                    second_importances, second_actual_sizes,
                                    threshold, known_num, l_max)
    if mode == Mode.invariants:
        return get_thresholded_task_invariants(first_importances, first_actual_sizes,
                                    second_importances, second_actual_sizes,
                                    threshold, known_num, l_max)
    
cdef get_thresholded_task_invariants(double[:, :] first_importances, int[:] first_actual_sizes,
                           double[:, :] second_importances, int[:] second_actual_sizes,
                           double threshold, int known_num, int l_max):
    
    ans = np.empty([known_num, 5], dtype = np.int32)
    
    raw_importances = np.empty([known_num])
    
    cdef int[:, :] ans_view = ans
    
    cdef int l, first_ind, second_ind, lambd
    cdef int pos = 0
   
    for l in range(l_max + 1):       
        for first_ind in range(first_actual_sizes[l]):
            for second_ind in range(second_actual_sizes[l]):
                if (first_importances[first_ind, l] * second_importances[second_ind, l] >= threshold):
                    ans_view[pos, 0] = first_ind
                    ans_view[pos, 1] = l
                    ans_view[pos, 2] = second_ind
                    ans_view[pos, 3] = l
                    ans_view[pos, 4] = 0
                    raw_importances[pos] = first_importances[first_ind, l] * second_importances[second_ind, l]
                    pos += 1
   
    return [ans[:pos], raw_importances[:pos]] 

cdef get_thresholded_task_covariants(double[:, :] first_importances, int[:] first_actual_sizes,
                           double[:, :] second_importances, int[:] second_actual_sizes,
                           double threshold, int known_num, int l_max):
    
    ans = np.empty([known_num, 5], dtype = np.int32)
    
    raw_importances = np.empty([known_num])
    
    cdef int[:, :] ans_view = ans
    
    cdef int l1, l2, first_ind, second_ind, lambd
    cdef int pos = 0
   
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for first_ind in range(first_actual_sizes[l1]):
                for second_ind in range(second_actual_sizes[l2]):
                    if (first_importances[first_ind, l1] * second_importances[second_ind, l2] >= threshold):
                        for lambd in range(abs_c(l1 - l2), min_c(l1 + l2, l_max) + 1):
                            ans_view[pos, 0] = first_ind
                            ans_view[pos, 1] = l1
                            ans_view[pos, 2] = second_ind
                            ans_view[pos, 3] = l2
                            ans_view[pos, 4] = lambd
                            raw_importances[pos] = first_importances[first_ind, l1] * second_importances[second_ind, l2]
                            pos += 1
   
    return [ans[:pos], raw_importances[:pos]]                       
                       

'''cpdef get_sorted(data):
    amplitudes = data.get_amplitudes()
    indices = np.empty(amplitudes.shape)
    invert_indices = np.empty(amplitudes.shape)

    new_covariants = np.empty(data.covariants_.shape)
    for lambd in range(amplitudes.shape[1]):
        indices[:data.actual_sizes_[lambd], lambd] = np.argsort(amplitudes[:data.actual_sizes_[lambd], lambd])
        new_covariants[:, :data.actual_sizes_[lambd], lambd, :(2 * lambd + 1)] = \
            data.covariants_[:, indices[:data.actual_sizes_[lambd], lambd], lambd, :(2 * lambd + 1)]


        invert_indices[indices[:data.actual_sizes_[lambd], lambd], lambd] = np.arange(data.actual_sizes_[lambd])

    return Data(new_covariants, data.actual_sizes_), invert_indices'''


cpdef sort_importances(importances, sizes):
    for lambd in range(importances.shape[1]):
        importances[:sizes[lambd], lambd] = np.sort(importances[:sizes[lambd], lambd])[::-1]
    return importances

cpdef get_thresholded_tasks(criterion, first_even, first_odd, second_even, second_odd, int desired_num, int l_max, mode_string):
    
    cdef Mode mode
    if mode_string == 'covariants':
        mode = Mode.covariants
    if mode_string == 'invariants':
        mode = Mode.invariants



    cdef double threshold_even
    cdef int num_even_even, num_odd_odd
    threshold_even, num_even_even, num_odd_odd = get_threshold(l_max, sort_importances(criterion(first_even), first_even.actual_sizes_),
                                                               first_even.actual_sizes_,
                                                               sort_importances(criterion(second_even), second_even.actual_sizes_), second_even.actual_sizes_,
                                                               sort_importances(criterion(first_odd), first_odd.actual_sizes_), first_odd.actual_sizes_,
                                                               sort_importances(criterion(second_odd), second_odd.actual_sizes_), second_odd.actual_sizes_,
                                                               desired_num, mode)

    cdef double threshold_odd
    cdef int num_even_odd, num_odd_even
    threshold_odd, num_even_odd, num_odd_even = get_threshold(l_max, sort_importances(criterion(first_even), first_even.actual_sizes_), first_even.actual_sizes_,
                                                              sort_importances(criterion(second_odd), second_odd.actual_sizes_), second_odd.actual_sizes_,
                                                              sort_importances(criterion(first_odd), first_odd.actual_sizes_), first_odd.actual_sizes_,
                                                              sort_importances(criterion(second_even), second_even.actual_sizes_), second_even.actual_sizes_,
                                                              desired_num, mode)        
      

    
    task_even_even = get_thresholded_task(criterion(first_even), first_even.actual_sizes_,
                                          criterion(second_even), second_even.actual_sizes_,
                                          threshold_even, num_even_even, l_max, mode)
    
    task_odd_odd = get_thresholded_task(criterion(first_odd), first_odd.actual_sizes_,
                                        criterion(second_odd), second_odd.actual_sizes_,
                                        threshold_even, num_odd_odd, l_max, mode)
    
    task_even_odd = get_thresholded_task(criterion(first_even), first_even.actual_sizes_,
                                         criterion(second_odd), second_odd.actual_sizes_,
                                         threshold_odd, num_even_odd, l_max, mode)
    
    task_odd_even = get_thresholded_task(criterion(first_odd), first_odd.actual_sizes_,
                                         criterion(second_even), second_even.actual_sizes_,
                                         threshold_odd, num_odd_even, l_max, mode)
    
    return task_even_even, task_odd_odd, task_even_odd, task_odd_even
                           
                           
                           
cdef get_threshold(int l_max, double[:, :] first_importances_1, int[:] first_actual_sizes_1,
                   double[:, :] second_importances_1, int[:] second_actual_sizes_1,
                   double[:, :] first_importances_2, int[:] first_actual_sizes_2,
                   double[:, :] second_importances_2, int[:] second_actual_sizes_2,
                   int desired_num, Mode mode, int min_iterations = 50):
    
    
    if (desired_num == -1):
        num_1_1 = get_total_num_full(l_max, first_importances_1, first_actual_sizes_1,
                                     second_importances_1, second_actual_sizes_1, -1.0, mode, Question.n_tasks)
        num_2_2 = get_total_num_full(l_max, first_importances_2, first_actual_sizes_2,
                                     second_importances_2, second_actual_sizes_2, -1.0, mode, Question.n_tasks)
        return -1.0, num_1_1, num_2_2
    
    cdef double left = -1.0
    cdef double first = get_upper_threshold(first_importances_1, first_actual_sizes_1, second_importances_1, second_actual_sizes_1, mode) + 1.0
    cdef double second = get_upper_threshold(first_importances_2, first_actual_sizes_2, second_importances_2, second_actual_sizes_2, mode) + 1.0
    
    cdef double right = fmax(first, second)
    cdef double middle = (left + right) / 2.0
    cdef int num_now, num_previous = -1
    cdef int num_it_no_change = 0
    while (True):
        middle = (left + right) / 2.0
        num_now = get_total_num_full(l_max, first_importances_1, first_actual_sizes_1, second_importances_1, second_actual_sizes_1, middle, mode, Question.n_pairs) + \
            get_total_num_full(l_max, first_importances_2, first_actual_sizes_2, second_importances_2, second_actual_sizes_2, middle, mode, Question.n_pairs)
        
        if (num_now == desired_num):
            left = middle
            break
        if (num_now > desired_num):
            left = middle
        if (num_now < desired_num):
            right = middle
            
        if (num_now == num_previous):
            num_it_no_change += 1
            if (num_it_no_change > min_iterations):
                break
        else:
            num_it_no_change = 0
        num_previous = num_now
            
    num_1_1 = get_total_num_full(l_max, first_importances_1, first_actual_sizes_1, second_importances_1, second_actual_sizes_1, left, mode, Question.n_tasks)
    num_2_2 = get_total_num_full(l_max, first_importances_2, first_actual_sizes_2, second_importances_2, second_actual_sizes_2, left, mode, Question.n_tasks)
    return left, num_1_1, num_2_2
    
    
cdef double get_upper_threshold(double[:, :] first_importances, int[:] first_actual_sizes, 
                             double[:, :] second_importances, int[:] second_actual_sizes, Mode mode):
    if mode == Mode.covariants:
        return get_upper_threshold_covariants(first_importances, first_actual_sizes, 
                                              second_importances, second_actual_sizes)
    if mode == Mode.invariants:
        return get_upper_threshold_invariants(first_importances, first_actual_sizes,
                                              second_importances, second_actual_sizes)
    
    
    
cdef double get_upper_threshold_invariants(double[:, :] first_importances, int[:] first_actual_sizes, 
                             double[:, :] second_importances, int[:] second_actual_sizes):
    cdef double ans = 0.0
    cdef int l
       
    for l in range(min_c(first_importances.shape[1], second_importances.shape[1])):      
        if (first_actual_sizes[l] > 0) and (second_actual_sizes[l] > 0):
            if (first_importances[0, l] * second_importances[0, l] > ans):
                ans = first_importances[0, l] * second_importances[0, l]
                    
    return ans


cdef double get_upper_threshold_covariants(double[:, :] first_importances, int[:] first_actual_sizes, 
                             double[:, :] second_importances, int[:] second_actual_sizes):
    cdef double ans = 0.0
    cdef int l1, l2
        
    cdef int second_size = second_importances.shape[1]
    for l1 in range(first_importances.shape[1]):
        for l2 in range(second_size):
            if (first_actual_sizes[l1] > 0) and (second_actual_sizes[l2] > 0):
                if (first_importances[0, l1] * second_importances[0, l2] > ans):
                    ans = first_importances[0, l1] * second_importances[0, l2]
                    
    return ans
                  
    
cdef int get_total_num_full(int l_max, double[:, :] first_importances, int[:] first_actual_sizes,
                            double[:, :] second_importances, int[:] second_actual_sizes,
                            double threshold, Mode mode, Question question):
    if mode == Mode.covariants:
        return get_total_num_full_covariants(l_max, first_importances, first_actual_sizes,
                                             second_importances, second_actual_sizes,
                                             threshold, question)
    if mode == Mode.invariants:
        return get_total_num_full_invariants(first_importances, first_actual_sizes,
                                             second_importances, second_actual_sizes,
                                             threshold)
    
cdef int get_total_num_full_invariants(double[:, :] first_importances, int[:] first_actual_sizes,
                            double[:, :] second_importances, int[:] second_actual_sizes,
                            double threshold):
    cdef int l
    cdef int second_size = second_importances.shape[1]
    cdef int res = 0
    for l in range(min_c(first_importances.shape[1], second_importances.shape[1])):
        if (first_actual_sizes[l] > 0) and (second_actual_sizes[l] > 0):
            res += get_total_num(first_importances[:first_actual_sizes[l], l],
                                 second_importances[:second_actual_sizes[l], l], threshold)
    return res

                                 
                                 
cdef int get_total_num_full_covariants(int l_max, double[:, :] first_importances, int[:] first_actual_sizes,
                            double[:, :] second_importances, int[:] second_actual_sizes,
                            double threshold, Question question):
    cdef int l1, l2
    cdef int second_size = second_importances.shape[1]
    cdef int res = 0
    cdef int now
    for l1 in range(first_importances.shape[1]):
        for l2 in range(second_size):
            if (first_actual_sizes[l1] > 0) and (second_actual_sizes[l2] > 0):
                now = get_total_num(first_importances[:first_actual_sizes[l1], l1],
                                     second_importances[:second_actual_sizes[l2], l2], threshold)
                if question == Question.n_pairs:
                    res += now
                else:
                    res += now *  (min_c(l1 + l2, l_max)  - abs_c(l1 - l2) + 1)
            
    return res
        
cdef int get_total_num(double[:] a, double[:] b, double threshold):
    cdef int b_size = b.shape[0]
    cdef int i, j, ans
    i = 0
    j = b_size
    ans = 0
    for i in range(a.shape[0]):
        while ((j > 0) and (a[i] * b[j - 1] < threshold)):
            j -= 1
        ans += j
    return ans
