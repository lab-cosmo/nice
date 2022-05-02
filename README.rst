.. role:: bash(code)
   :language: bash

NICE
====

NICE (N-body Iteratively Contracted Equivariants) is a set of tools designed for the calculation of 
invariant and covariant atomic structure representations. It allows for
automatic selection of the most informative combinations of high order spectrum elements
and performs their efficient computation using recurrence relations. 

Although it is designed specifically for atomistic machine learning, NICE in principle 
can be applied to other machine learning tasks, such as those which involve signals in a ball or on a sphere, all which require invariant or covariant outputs. 

++++++++++++
Installation
++++++++++++

1. Install `librascal <https://github.com/cosmo-epfl/librascal>`_
2. git clone or download archive with nice and unpack
3. cd to root nice directory and run :bash:`pip3 install .`

+++++++++++++
Documentation
+++++++++++++

Documentation can be found `here <https://lab-cosmo.github.io/nice/>`_

++++++++++
References
++++++++++

If you are using NICE, please cite `this article <https://aip.scitation.org/doi/10.1063/5.0021116>`_. 

[1] Jigyasa Nigam, Sergey Pozdnyakov, and Michele Ceriotti. "Recursive evaluation and iterative contraction of N-body equivariant features." The Journal of Chemical Physics 153.12 (2020): 121101.
