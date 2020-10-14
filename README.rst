.. role:: bash(code)
   :language: bash

NICE
====

NICE (N-body Iteratively Contracted Equivariants) is a set of tools designed for calculation of 
invariant and covariant atomic structure representations. It allows to
automatically select most informative combinations of high order spectrum elements
and perform their efficient computation using recurrency relations. 

Though being designed specifically for atomistic machine learning, NICE in principle 
can be applied to other machine learning tasks which involves signals in a ball or on a sphere   with necessasity to produce invariant or covariant output. 

++++++++++++
Installation
++++++++++++

1. Install `librascal <https://github.com/cosmo-epfl/librascal>`_
2. git clone or download archive with nice and unpack
3. cd to root nice directory and run :bash:`pip3 install .`

+++++++++++++
Documentation
+++++++++++++

Documentation is `here <https://serfg.github.io/nice/>`_

++++++++++
References
++++++++++

If you are using NICE please cite `this article <https://aip.scitation.org/doi/10.1063/5.0021116>`_. 

[1] Nigam, Jigyasa, Sergey Pozdnyakov, and Michele Ceriotti. "Recursive evaluation and iterative contraction of N-body equivariant features." The Journal of Chemical Physics 153.12 (2020): 121101.
