============
Introduction
============
There are two command line tools that can implement the NICE sequence to output features. Details of the theory behind the NICE sequence can be read "`here <https://serfg.github.io/nice/theory.html>`_". The first tool takes inputs from users for fitting a model using the NICE sequence. The output NICE model from first tool can be input by the second tool to predict features for the dataset. An example to achieve this using a "`methane <https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528>`_ database" is given below:

.. code:: python
        
    wget "https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528" -O methane.extxyz.gz
    gunzip -v methane.extxyz.gz
    

Now, that we have the database downloaded and unzipped in the same folder, we can use it as an input. 

*fitting_nice.py*
-----------------

To get a fitted model for the given "methane.extxyz" database, we need to provide the following inputs:

    1. The database file.
    2. The name for final output file. *(-o)*
    3. The type of NICE model to be fitted *(-w)*. It can be:
        a. A different NICE model centered around each specie (1)
        b. A single NICE model irrespective of the specie (2)
    4. Index for ase.io.read commands *(--train_subset)*. This is the subset of the database used for training. 
    5. The number of environments to fit nice transfomers *(--environments_for_fitting)*.
    6. Input for spherical expansion parameters *(--interaction_cutoff, --max_radial, --max_angular, --gaussian_sigma_constant)*. Keeping 'gaussian_sigma_type': 'Constant','cutoff_smooth_width': 0.3, and 'radial_basis': 'GTO'
    7. Input for standardblocks for covariants (cov) and invariants (inv). These can be in form of a single value for all blocks or in the form of a list of different values for each block. 
        *Number of the most important input pairs for expansion *(--numexpcov, --numexpinv)*
        *Number of features to be considered for purification step *(--maxtakecov, --maxtakeinv)*
        *Number of components for the PCA step *(--ncompcov, --ncompinv)*
    8. The desired number of blocks in the StandardBlocks *(--blocks)*.
    9. Any additional hypers *(--json)*

So, the input can be given as:

.. code-block:: python
    
    python3 fitting_nice.py methane.extxyz -o nice_output -w 1 --train_subset "0:10000" --environments_for_fitting 1000 --interaction_cutoff 6.3 --max_radial 5 --max_angular 5 --gaussian_sigma_constant 0.05 --numexpcov 150 --numexpinv 300 --maxtakecov 10 --maxtakeinv 50 --ncompcov 50 --ncompinv 200 --blocks 4
    

Note that here the StandardBlocks inputs are in form of a single value. 
This is equivalent to:

.. code-block:: python
    
    python3 fitting_nice.py methane.extxyz -o nice_output -w 1 --train_subset "0:10000" --environments_for_fitting 1000 --interaction_cutoff 6.3 --max_radial 5 --max_angular 5 --gaussian_sigma_constant 0.05 --numexpcov "150,150," --numexpinv "300,300,300" --maxtakecov "10,10," --maxtakeinv "50,50,50" --ncompcov "50,50," --ncompinv "200,200,200" --blocks 4
    
Here, the StandardBlocks inputs are in form of a list. The last block only considers invariants irrespective of the user entry. 

This means that we are forming the following sequence for NICE framework:

.. code-block:: python

    StandardSequence([
        StandardBlock(ThresholdExpansioner(num_expand=150),
                      CovariantsPurifierBoth(max_take=10),
                      IndividualLambdaPCAsBoth(n_components=50),
                      ThresholdExpansioner(num_expand=300, mode='invariants'),
                      InvariantsPurifier(max_take=50),
                      InvariantsPCA(n_components=200)),
        StandardBlock(ThresholdExpansioner(num_expand=150),
                      CovariantsPurifierBoth(max_take=10),
                      IndividualLambdaPCAsBoth(n_components=50),
                      ThresholdExpansioner(num_expand=300, mode='invariants'),
                      InvariantsPurifier(max_take=50),
                      InvariantsPCA(n_components=200)),
        StandardBlock(None, None, None,
                      ThresholdExpansioner(num_expand=300, mode='invariants'),
                      InvariantsPurifier(max_take=50),
                      InvariantsPCA(n_components=200))
    ],
                            initial_scaler=InitialScaler(
                                mode='signal integral', individually=True))




Here, *-w* is 1, so this sequence runs for all species. In this example, since there are two species - H (Specie index 1), and C (Specie index 6), so the sequence runs twice.

The output of this command line tool is:
    1. Spherical expansion parameters including those input by the user
    
    .. code-block:: python
    
        HYPERS = {
            'interaction_cutoff': interaction_cutoff,
            'max_radial': max_radial,
            'max_angular': max_angular,
            'gaussian_sigma_type': 'Constant',
            'gaussian_sigma_constant': gaussian_sigma_constant,
            'cutoff_smooth_width': 0.3,
            'radial_basis': 'GTO'}

    2. Fitted NICE model


*transforming_nice.py*
----------------------

To get the predicted features for the same "methane.extxyz" database, we need to provide the following inputs:

    1. The database file.
    2. The name for final output file. *(-o)*
    3. Index for ase.io.read commands *(--index)*. This is the subset of the database of which the features will be output.
    4. Output from *fitting_nice.py*, i.e. parameters for spherical expansion, and the fitted NICE model. *(--nice)*

User can input these like:


.. code-block:: python
    
    python3 transforming_nice.py methane.extxyz -o out_transform --index "10000:15000" --nice nice_output

Here, the subset of the database undergoes spherical expansion with the same parameters as those used for the training dataset provided from the output of previous *fitting_nice.py* stored in *"nice_output"*. Then to predit features, the NICE model output from *"nice_output"* is used. It is to be noted that here the predicted features are for each specie in the dataset.  

For the dataset, the output of this command line tool are the prediceted:
    1. Features,
    2. Compositional features, and
    3. Energies,
in the same output file specified earlier *"out_transform"*. Now these features are ready to be used.

*Relative Error*
----------------

To get the relative error, we can define the following functions in a new python notebook:

.. code-block:: python

    def get_rmse(first, second):
    return np.sqrt(np.mean((first - second) ** 2))
    
    def get_standard_deviation(values):
    return np.sqrt(np.mean((values - np.mean(values)) ** 2))
    
    def get_relative_performance(predictions, values):
    return get_rmse(predictions, values) / get_standard_deviation(values)
    
    def estimate_performance(clf, data_train, data_test, targets_train, targets_test):
    clf.fit(data_train, targets_train)
    return get_relative_performance(clf.predict(data_test), targets_test)

Now, for the given example, we need the training and testing (or predicted) features, and energies. 

    1. To get the NICE model that can be used to get the features, we run:

    .. code-block:: python

        python3 fitting_nice.py methane.extxyz -o nice_output -w 1 --train_subset "0:10000" --environments_for_fitting 1000 --interaction_cutoff 6.3 --max_radial 5 --max_angular 5 --gaussian_sigma_constant 0.05 --numexpcov 150 --numexpinv 300 --maxtakecov 10 --maxtakeinv 50 --ncompcov 50 --ncompinv 200 --blocks 4

    2. Then to get the train_features, we use the output of this on the same subset of the methane.extxyz database.

    .. code-block:: python

        python3 transforming_nice.py methane.extxyz -o out_transform_train --index "0:10000" --nice nice_output

    In the same notebook, we can:

    3. And, for the test_features, we use the output of *1* again as done in *2*, for a diffrent subset of the database.

    .. code-block:: python

        python3 transforming_nice.py methane.extxyz -o out_transform_train --index "10000:15000" --nice nice_output

    4. Note that in this database, the energies of the structure are defined and can be directly sourced. HARTREE_TO_EV is defined as *27.211386245988*

    .. code-block:: python

        train_energies = [structure.info['energy'] for structure in train_structures]
        train_energies = np.array(train_energies) * HARTREE_TO_EV
    
        test_energies = [structure.info['energy'] for structure in test_structures]
        test_energies = np.array(test_energies) * HARTREE_TO_EV

    5. Now running the defined functions,

    .. code-block:: python

        errors = []
        for el in tqdm.tqdm(grid):
            errors.append(estimate_performance(BayesianRidge(), train_features[:el],
                                       test_features, train_energies[:el],
                                       test_energies))

        print(errors)
        
        from matplotlib import pyplot as plt
        plt.plot(grid, errors, 'bo')
        plt.plot(grid, errors, 'b')
        plt.xlabel("number of structures")
        plt.ylabel("relative error")
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

This gives the following relative error plot as a function of number of structures: 