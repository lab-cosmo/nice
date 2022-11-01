.. code:: ipython3

    # downloading dataset from https://archive.materialscloud.org/record/2020.110
    
    !wget "https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528" -O methane.extxyz.gz
    !gunzip -k methane.extxyz.gz
    
    import numpy as np
    import ase.io
    import tqdm
    from nice.blocks import *
    from nice.utilities import *
    from matplotlib import pyplot as plt
    from sklearn.linear_model import BayesianRidge
    
    structures = ase.io.read('methane.extxyz', index='0:1000')
    
    HYPERS = {
        'interaction_cutoff': 6.3,
        'max_radial': 5,
        'max_angular': 5,
        'gaussian_sigma_type': 'Constant',
        'gaussian_sigma_constant': 0.05,
        'cutoff_smooth_width': 0.3,
        'radial_basis': 'GTO'
    }
    
    all_species = get_all_species(structures)
    
    coefficients = get_spherical_expansion(structures, HYPERS, all_species)
