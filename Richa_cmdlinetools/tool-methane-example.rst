Introduction
============
There are two command line tools that can implement the NICE sequence to output features. Details of the theory behind the NICE sequence can be read `here <https://serfg.github.io/nice/theory.html>_`. The first tool takes inputs from users for fitting a model using the NICE sequence. The output NICE model from first tool can be input by the second tool to predict features for the dataset. An example to achieve this using a "`methane <https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528>`_ database" is given below.

.. code-block:: python
    wget "https://archive.materialscloud.org/record/file?file_id=b612d8e3-58af-4374-96ba-b3551ac5d2f4&filename=methane.extxyz.gz&record_id=528" -O methane.extxyz.gz
    gunzip -v methane.extxyz.gz
    

Now, that we have the database downloaded and unzipped in the same folder, we can use it as an input. 

fitting_nice.py
---------------

To get a fitted model for the given "methane.extxyz" database, we need to provide the following inputs:
1. The database file.
2. The name for final output file. *(-o)*
3. The type of NICE model to be fitted *(-w)*. It can be:
    *A different NICE model centered around each specie (1)
    *A single NICE model irrespective of the specie (2)
3. Index for training commands for ase.io.read commands *(--train_subset)*. This is the subset of the database used for training. 
4. The number of environments to fit nice transfomers *(--environments_for_fitting)*.
5. Input for HYPERS parameters *(--interaction_cutoff, --max_radial, --max_angular, --gaussian_sigma_constant)*. Keeping 'gaussian_sigma_type': 'Constant','cutoff_smooth_width': 0.3, and 'radial_basis': 'GTO'
6. Input for standardblocks for covariants (cov) and invariants (inv). These can be in form of a single value for all blocks or in the form of a list of different values for each block. 
    *Number of the most important input pairs for expansion *(--numexpcov, --numexpinv)*
    *Number of features to be considered for purification step *(--maxtakecov, --maxtakeinv)*
    *Number of components for the PCA step *(--ncompcov, --ncompinv)*
7. The desired number of blocks in the StandardBlocks *(--blocks)*.
8. Any additional hypers *(--json)*

So, the input can be given as:

.. code-block:: python
    python3 fitting_nice.py methane.extxyz -o nice_output -w 1 --train_subset "0:10000" --environments_for_fitting 1000 --interaction_cutoff 6.3 --max_radial 5 --max_angular 5 --gaussian_sigma_constant 0.05 --numexpcov 150 --numexpinv 300 --maxtakecov 10 --maxtakeinv 50 --ncompcov 50 --ncompinv 200 --blocks 4
    

Note that here the StandardBlocks inputs are in form of a single value. This is equivalent to:

.. code-block:: python
    python3 fitting_nice.py methane.extxyz -o nice_output -w 1 --train_subset "0:10000" --environments_for_fitting 1000 --interaction_cutoff 6.3 --max_radial 5 --max_angular 5 --gaussian_sigma_constant 0.05 --numexpcov "150,150," --numexpinv "300,300,300" --maxtakecov "10,10," --maxtakeinv "50,50,50" --ncompcov "50,50," --ncompinv "200,200,200" --blocks 4
    
Here, the StandardBlocks inputs are in form of a list. The last block only considers invariants irrespective of the user entry.
