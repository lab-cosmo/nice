from libc.math cimport sin, M_PI, sqrt
cimport cython
import numpy as np

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
    cdef bint swap_now
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
                
        if (mu == 0):
            swap_now = (imag_now * imag_now > real_now * real_now)
            
        if (swap_now):
            tmp = real_now
            real_now = -imag_now
            imag_now = tmp
        
        if (mu > 0):
            ans_placeholder[mu + lambd] = real_now
            ans_placeholder[-mu + lambd] = imag_now
        else:
            ans_placeholder[lambd] = real_now / sqrt_2
            
            
cpdef do_full_expansion(double[:, :, :, :, :] clebsh_gordan,
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
                            
    return new_same_parity_features, new_other_parity_features