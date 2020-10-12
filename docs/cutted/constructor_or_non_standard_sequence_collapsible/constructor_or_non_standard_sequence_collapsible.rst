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
    
    structures = ase.io.read('methane.extxyz', 
                             index = '0:1000')
    
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



.. parsed-literal::

    --2020-10-12 21:26:57--  https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528
    Resolving archive.materialscloud.org (archive.materialscloud.org)... 148.187.96.41
    Connecting to archive.materialscloud.org (archive.materialscloud.org)|148.187.96.41|:443... connected.
    HTTP request sent, awaiting response... 302 FOUND
    Location: https://object.cscs.ch/archive/b6/12/d8e3-58af-4374-96ba-b3551ac5d2f4/data?response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3B%20filename%3Dmethane.extxyz.gz&Expires=1602530877&Signature=VYUS8wL0D0Oadx%2BwOI4W57%2BAO5Q%3D&AWSAccessKeyId=ee64314446074ed3ab5f375a522a4893 [following]
    --2020-10-12 21:26:57--  https://object.cscs.ch/archive/b6/12/d8e3-58af-4374-96ba-b3551ac5d2f4/data?response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3B%20filename%3Dmethane.extxyz.gz&Expires=1602530877&Signature=VYUS8wL0D0Oadx%2BwOI4W57%2BAO5Q%3D&AWSAccessKeyId=ee64314446074ed3ab5f375a522a4893
    Resolving object.cscs.ch (object.cscs.ch)... 148.187.25.200, 148.187.25.201, 148.187.25.202
    Connecting to object.cscs.ch (object.cscs.ch)|148.187.25.200|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1218139661 (1.1G) [application/octet-stream]
    Saving to: ‘methane.extxyz.gz’
    
    methane.extxyz.gz   100%[===================>]   1.13G  21.8MB/s    in 37s     
    
    2020-10-12 21:27:34 (31.5 MB/s) - ‘methane.extxyz.gz’ saved [1218139661/1218139661]
    


.. parsed-literal::

    100%|██████████| 10/10 [00:00<00:00, 29.15it/s]
    100%|██████████| 2/2 [00:00<00:00, 162.26it/s]

