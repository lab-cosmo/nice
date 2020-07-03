cdef void single_contraction(double[:, :, :, :, :] clebsh_gordan,
                            double[:] first_covariant, int l1,
                            double[:] second_covariant, int l2,
                            int lambd, double[:] ans_placeholder, 
                            double[:, :] buff)


cdef int min_c(int a, int b)
cdef int max_c(int a, int b)
    
cdef int abs_c(int a)


