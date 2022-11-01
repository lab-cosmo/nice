.. code:: ipython3

    # cell to wrap in collapsible in future
    
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
    
    HARTREE_TO_EV = 27.211386245988
    train_subset = "0:10000"  #input for ase.io.read command
    test_subset = "10000:15000"  #input to ase.io.read command
    environments_for_fitting = 1000  #number of environments to fit nice transfomers
    grid = [150, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500,
            10000]  #for learning curve
    
    #HYPERS for librascal spherical expansion coefficients
    HYPERS = {
        'interaction_cutoff': 6.3,
        'max_radial': 5,
        'max_angular': 5,
        'gaussian_sigma_type': 'Constant',
        'gaussian_sigma_constant': 0.05,
        'cutoff_smooth_width': 0.3,
        'radial_basis': 'GTO'
    }
    
    
    #our model:
    def get_nice():
        return StandardSequence([
            StandardBlock(ThresholdExpansioner(num_expand=150),
                          CovariantsPurifierBoth(max_take=10),
                          IndividualLambdaPCAsBoth(n_components=50),
                          ThresholdExpansioner(num_expand=300, mode='invariants'),
                          InvariantsPurifier(max_take=50),
                          InvariantsPCA(n_components=200)),
            StandardBlock(ThresholdExpansioner(num_expand=150),
                          CovariantsPurifierBoth(max_take=10),
                          IndividualLambdaPCAsBoth(n_components=10),
                          ThresholdExpansioner(num_expand=300, mode='invariants'),
                          InvariantsPurifier(max_take=50),
                          InvariantsPCA(n_components=200)),
            StandardBlock(ThresholdExpansioner(num_expand=150),
                          CovariantsPurifierBoth(max_take=10), None,
                          ThresholdExpansioner(num_expand=300, mode='invariants'),
                          InvariantsPurifier(max_take=50),
                          InvariantsPCA(n_components=200))
        ],
                                initial_scaler=InitialScaler(
                                    mode='signal integral', individually=True))
    
    
    train_structures = ase.io.read('methane.extxyz', index=train_subset)
    
    test_structures = ase.io.read('methane.extxyz', index=test_subset)
    
    all_species = get_all_species(train_structures + test_structures)
    
    train_coefficients = get_spherical_expansion(train_structures, HYPERS,
                                                 all_species)
    
    test_coefficients = get_spherical_expansion(test_structures, HYPERS,
                                                all_species)
    
    #individual nice transformers for each atomic specie in the dataset
    nice = {}
    for key in train_coefficients.keys():
        nice[key] = get_nice()
    
    for key in train_coefficients.keys():
        nice[key].fit(train_coefficients[key][:environments_for_fitting])
