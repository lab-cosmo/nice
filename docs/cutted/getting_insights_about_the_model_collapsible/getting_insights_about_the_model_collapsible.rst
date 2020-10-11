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
    train_subset = "0:10000"    #input for ase.io.read command
    test_subset = "10000:15000"     #input to ase.io.read command
    environments_for_fitting = 1000    #number of environments to fit nice transfomers
    grid =  [150, 200, 350, 500, 750, 1000,
             1500, 2000, 3000, 5000, 7500, 10000] #for learning curve
    
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
    def get_transformer():
        return StandardSequence([StandardBlock(ThresholdExpansioner(num_expand = 150),
                                                  CovariantsPurifierBoth(max_take = 10),
                                                      IndividualLambdaPCAsBoth(n_components = 50),
                                                     ThresholdExpansioner(num_expand =300, mode = 'invariants'),
                                                     InvariantsPurifier(max_take = 50),
                                                      InvariantsPCA(n_components = 200)),
                                 StandardBlock(ThresholdExpansioner(num_expand = 150),
                                                  CovariantsPurifierBoth(max_take = 10),
                                                      IndividualLambdaPCAsBoth(n_components = 50),
                                                     ThresholdExpansioner(num_expand =300, mode = 'invariants'),
                                                     InvariantsPurifier(max_take = 50),
                                                      InvariantsPCA(n_components = 200)),
                                StandardBlock(ThresholdExpansioner(num_expand = 150),
                                                  CovariantsPurifierBoth(max_take = 10),
                                                      IndividualLambdaPCAsBoth(n_components = 50),
                                                      ThresholdExpansioner(num_expand =300, mode = 'invariants'),
                                                  InvariantsPurifier(max_take = 50),
                                                     InvariantsPCA(n_components = 200))
                                       ],
                                initial_scaler = InitialScaler(mode = 'signal integral',
                                                               individually = True)
                              )
    
    
    train_structures = ase.io.read('methane.extxyz', 
                             index = train_subset)
    
    test_structures = ase.io.read('methane.extxyz', 
                             index = test_subset)
    
    all_species = get_all_species(train_structures + test_structures)
    
    train_coefficients = get_spherical_expansion(train_structures, HYPERS, all_species)
    
    
    
    test_coefficients = get_spherical_expansion(test_structures, HYPERS, all_species)
    
    #individual transformers for each atomic specie in dataset
    transformers = {}
    for key in train_coefficients.keys():
        transformers[key] = get_transformer()
        
    for key in train_coefficients.keys():
        transformers[key].fit(train_coefficients[key][:environments_for_fitting])


.. parsed-literal::

    Will not apply HSTS. The HSTS database must be a regular and non-world-writable file.
    ERROR: could not open HSTS store at '/home/pozdn/.wget-hsts'. HSTS will be disabled.
    --2020-10-08 05:12:54--  https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528
    Resolving archive.materialscloud.org (archive.materialscloud.org)... 148.187.96.41
    Connecting to archive.materialscloud.org (archive.materialscloud.org)|148.187.96.41|:443... connected.
    HTTP request sent, awaiting response... 302 FOUND
    Location: https://object.cscs.ch/archive/b6/12/d8e3-58af-4374-96ba-b3551ac5d2f4/data?response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3B%20filename%3Dmethane.extxyz.gz&Expires=1602126834&Signature=yMiITsP7kQ7Z7zbPsNkPExbHI3Q%3D&AWSAccessKeyId=ee64314446074ed3ab5f375a522a4893 [following]
    --2020-10-08 05:12:54--  https://object.cscs.ch/archive/b6/12/d8e3-58af-4374-96ba-b3551ac5d2f4/data?response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3B%20filename%3Dmethane.extxyz.gz&Expires=1602126834&Signature=yMiITsP7kQ7Z7zbPsNkPExbHI3Q%3D&AWSAccessKeyId=ee64314446074ed3ab5f375a522a4893
    Resolving object.cscs.ch (object.cscs.ch)... 148.187.25.202, 148.187.25.200, 148.187.25.201
    Connecting to object.cscs.ch (object.cscs.ch)|148.187.25.202|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1218139661 (1.1G) [application/octet-stream]
    Saving to: ‘methane.extxyz.gz’
    
    methane.extxyz.gz   100%[===================>]   1.13G   111MB/s    in 19s     
    
    2020-10-08 05:13:13 (62.1 MB/s) - ‘methane.extxyz.gz’ saved [1218139661/1218139661]
    


.. parsed-literal::

    In /home/pozdn/.local/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib/_classic_test.mplstyle: 
    The text.latex.preview rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
    In /home/pozdn/.local/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib/_classic_test.mplstyle: 
    The mathtext.fallback_to_cm rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
    In /home/pozdn/.local/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib/_classic_test.mplstyle: Support for setting the 'mathtext.fallback_to_cm' rcParam is deprecated since 3.3 and will be removed two minor releases later; use 'mathtext.fallback : 'cm' instead.
    In /home/pozdn/.local/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib/_classic_test.mplstyle: 
    The validate_bool_maybe_none function was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
    In /home/pozdn/.local/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib/_classic_test.mplstyle: 
    The savefig.jpeg_quality rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
    In /home/pozdn/.local/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib/_classic_test.mplstyle: 
    The keymap.all_axes rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
    In /home/pozdn/.local/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib/_classic_test.mplstyle: 
    The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
    In /home/pozdn/.local/lib/python3.6/site-packages/matplotlib/mpl-data/stylelib/_classic_test.mplstyle: 
    The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
    100%|██████████| 100/100 [00:00<00:00, 105.48it/s]
    100%|██████████| 2/2 [00:00<00:00, 24.54it/s]
    100%|██████████| 50/50 [00:00<00:00, 102.47it/s]
    100%|██████████| 2/2 [00:00<00:00, 46.46it/s]
    /home/pozdn/.local/lib/python3.6/site-packages/nice/blocks/compressors.py:201: UserWarning: Amount of provided data is less than the desired one to fit PCA. Number of components is 200, desired number of environments is 2000, actual number of environments is 1000.
      self.n_components, num_fit_now, X.shape[0]))
    /home/pozdn/.local/lib/python3.6/site-packages/nice/blocks/compressors.py:201: UserWarning: Amount of provided data is less than the desired one to fit PCA. Number of components is 200, desired number of environments is 2000, actual number of environments is 1000.
      self.n_components, num_fit_now, X.shape[0]))
    /home/pozdn/.local/lib/python3.6/site-packages/nice/blocks/compressors.py:201: UserWarning: Amount of provided data is less than the desired one to fit PCA. Number of components is 200, desired number of environments is 2000, actual number of environments is 1000.
      self.n_components, num_fit_now, X.shape[0]))
    /home/pozdn/.local/lib/python3.6/site-packages/nice/blocks/compressors.py:201: UserWarning: Amount of provided data is less than the desired one to fit PCA. Number of components is 200, desired number of environments is 2000, actual number of environments is 1000.
      self.n_components, num_fit_now, X.shape[0]))
    /home/pozdn/.local/lib/python3.6/site-packages/nice/blocks/compressors.py:201: UserWarning: Amount of provided data is less than the desired one to fit PCA. Number of components is 200, desired number of environments is 2000, actual number of environments is 1000.
      self.n_components, num_fit_now, X.shape[0]))
    /home/pozdn/.local/lib/python3.6/site-packages/nice/blocks/compressors.py:201: UserWarning: Amount of provided data is less than the desired one to fit PCA. Number of components is 200, desired number of environments is 2000, actual number of environments is 1000.
      self.n_components, num_fit_now, X.shape[0]))

