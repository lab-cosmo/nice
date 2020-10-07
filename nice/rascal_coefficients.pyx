import numpy as np
cimport cython
from nice_utilities cimport single_contraction, min_c, abs_c, max_c
from libc.math cimport sin, M_PI, sqrt, fmax
import tqdm
import rascal
from ase import Atoms
from rascal.representations import SphericalInvariants as SOAP
from rascal.representations import SphericalExpansion as SPH
from rascal.neighbourlist.structure_manager import (
        mask_center_atoms_by_species, mask_center_atoms_by_id)
import warnings
import copy
from multiprocessing import Pool, cpu_count


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void copy_coefs(double[:, :, :, :] coefficients, int[:] central_species, int central_now,
                double[:, :, :, :] ans):
    cdef int now = 0
    cdef int n_radial = coefficients.shape[1]
    cdef int l_max = coefficients.shape[2] - 1
    cdef int env_ind, radial_ind, l, m
    
    
    for env_ind in range(coefficients.shape[0]):
        if central_species[env_ind] == central_now:
            for radial_ind in range(n_radial):
                for l in range(l_max + 1):
                    for m in range(2 * l_max + 1):
                        ans[now, radial_ind, l, m] = coefficients[env_ind, radial_ind, l, m]
            now += 1


def split_by_central_specie(all_species, species, coefficients, show_progress = True): 
    result = {}
    for specie in tqdm.tqdm(species, disable = not show_progress):
        num_now = np.sum(all_species == specie)
        result[specie] = np.empty([num_now, coefficients.shape[1], coefficients.shape[2], coefficients.shape[3]])
        copy_coefs(coefficients, all_species, specie, result[specie])
    return result   
    


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef convert_rascal_coefficients(double[:, :] coefficients, int n_max, int n_types, int l_max):
    cdef int n_envs = coefficients.shape[0]
    cdef int env_ind, n, l, m
    cdef int n_radial = n_max * n_types
    cdef int now
    ans = np.zeros([n_envs, n_radial, l_max + 1, 2 * l_max + 1])
    cdef double[:, :, :, :] ans_view = ans
    
    for env_ind in range(n_envs):
        now = 0
        for n in range(n_radial):
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    ans_view[env_ind, n, l, m + l] = coefficients[env_ind, now]
                    now += 1
    return ans

   

def process_structures(structures, delta = 0.1):   
    """Satisfying librascal desire of having all atoms 
    inside the cell even if structure is not periodic. 
    (changes only non periodic structures)
    """

    result = []
    for structure in structures:
        if True in structure.pbc:
            result.append(copy.deepcopy(structure))
        else:
            current = copy.deepcopy(structure)
            for dim in range(3):
                min_now = np.min( current.positions[:, dim])
                current.positions[:, dim] =  current.positions[:, dim] - min_now + delta
            
            spreads = []
            for dim in range(3):                
                spreads.append(np.max(current.positions[:, dim]) + delta)
            current.cell = spreads
            result.append(current)
    return result


def get_rascal_coefficients(structures, HYPERS, n_types):
   
    
    sph = SPH(**HYPERS)
    try:
        n_max = HYPERS['max_radial']
        l_max = HYPERS['max_angular']
    except KeyError:
        raise KeyError("max_radial and max_angular should be specified")
        
    structures = process_structures(structures)
    
    feat = sph.transform(structures).get_features(sph)
    res = convert_rascal_coefficients(feat, n_max, n_types, l_max)
    
    #if (normalize):
    #    normalize_by_ps(res)
    return np.array(res)


def get_rascal_coefficients_stared(task):
    return get_rascal_coefficients(*task)


            
