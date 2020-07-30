cdef void single_contraction(const double[:, :, :, :, :] clebsh_gordan,
                            double* first_covariant, int l1,
                            double* second_covariant, int l2,
                            int lambd, double* ans_placeholder, 
                            double** buff) nogil


cdef int min_c(int a, int b) nogil
cdef int max_c(int a, int b) nogil
    
cdef int abs_c(int a) nogil


