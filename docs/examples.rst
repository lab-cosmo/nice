Examples
========
Among the other things this repository contains examples folder. 

qm9_home_pc.ipynb and qm9_small.ipynb might become a good starting point for 
calculations as a code snippets. 

methane_small.ipynb, methane_medium.ipynb and qm9_small.ipynb provide general advice on
appropriate real life hyperparameters. 

QM9 dataset is `available  <https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904>`_ 
in the form of separate .xyz files for each molecule in a such special format
that it can not be read by `ase <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_.

First cells of qm9_home_pc.ipynb and qm9_small.ipynb notebooks contain code to borrow which
fetches raw QM9 dataset and parses it into single ase .extxyz file. 
