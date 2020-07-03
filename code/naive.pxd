cpdef double compute_bispectrum_single(double[:, :, :, :, :] clebsh_gordan, double[:] c1,
                   double[:] c2, double[:] c3, int l1, int l2, int l, double[:, :] buff)

cpdef double[:] compute_bispectrum(double[:, :, :, :, :] clebsh_gordan, double[:, :, :] c, 
                         int l_max, bint only_even = ?)

cpdef double compute_powerspectrum_single(double[:] c1, double[ :] c2, int l)

cpdef double[:] compute_powerspectrum(double[:, :, :] c, int l_max)
 
    

                
    