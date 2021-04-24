import numpy as np
import ase.io
from nice.blocks import *
from nice.utilities import *
from nice.rascal_coefficients import process_structures
import copy
from rascal.representations import SphericalInvariants

def get_nice_powerspectrum():
    return StandardSequence([       
        StandardBlock(None, None, None,
                      ThresholdExpansioner(num_expand=None, mode='invariants'),
                      None, None)
    ],
                            initial_scaler=None)

def get_nice_ps_kernel(structures, hypers):
    
    all_species = get_all_species(structures)
    coefficients = get_spherical_expansion(structures, hypers, all_species, split_by_central_specie=False,
                                           show_progress = False)
    nice = get_nice_powerspectrum()
    nice.fit(coefficients)
    nice_ps = nice.transform(coefficients, return_only_invariants = True)[2]
    return nice_ps.dot(nice_ps.T)

def get_rascal_ps_kernel(structures, hypers):
    structures = process_structures(structures)
    soap = SphericalInvariants(**hypers)
    librascal_ps = soap.transform(structures).get_features(soap)
    return librascal_ps.dot(librascal_ps.T)

def test_powerspectrum_kernels(epsilon = 1e-10):
    structures = ase.io.read('../reference_configurations/methane_100.extxyz', index = ':')
    HYPERS = {
        'interaction_cutoff': 6.3,
        'max_radial': 5,
        'max_angular': 5,
        'gaussian_sigma_type': 'Constant',
        'gaussian_sigma_constant': 0.3,
        'cutoff_smooth_width': 0.3,
        'radial_basis': 'GTO',

    }

    HYPERS_PS = copy.deepcopy(HYPERS)
    HYPERS_PS['normalize'] = False
    HYPERS_PS['soap_type'] = 'PowerSpectrum'
    
    nice_kernel = get_nice_ps_kernel(structures, HYPERS)
    rascal_kernel = get_rascal_ps_kernel(structures, HYPERS_PS)
    
    nice_kernel = np.reshape(nice_kernel, [-1])
    rascal_kernel = np.reshape(rascal_kernel, [-1])
    
    mask = rascal_kernel > epsilon
    nice_kernel = nice_kernel[mask]
    rascal_kernel = rascal_kernel[mask]
    
    ratios = nice_kernel / rascal_kernel
    discrepancy = (np.max(ratios) - np.min(ratios)) / np.mean(ratios)   
    assert discrepancy < epsilon
    
    