Real-Life Examples of Nice
==========================


Among other things, this repository contains scripts and notebooks to contextualize NICE into real-world problems. These examples are similar to the procedures reported in Jigyasa Nigam, Sergey Pozdnyakov, and Michele Ceriotti. "Recursive evaluation and iterative contraction of N-body equivariant features." The Journal of Chemical Physics 153.12 (2020): 121101, but not direct productions.

In `qm9_home_pc.ipynb` and `qm9_small.ipynb` construct similar machine learned potentials for the QM9 dataset (see below). `qm9_home_pc.ipynb` is intended to run on a local workstation, whereas `qm9_small.ipynb` is best suited for HPC resources. We have also provided examples for the methane dataset (https://archive.materialscloud.org/record/2020.110). All notebooks include general advice on
appropriate real-life hyperparameters. 

QM9 dataset is `available  <https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904>`_ 
in the form of separate .xyz files for each molecule in such a special format
that it can not be read by `ase <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_.
The first cells of qm9_home_pc.ipynb and qm9_small.ipynb notebooks contain code to fetch the raw QM9 dataset and parses it into a single ase .extxyz file. 
