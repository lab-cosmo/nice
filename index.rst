.. NICE documentation master file, created by
   sphinx-quickstart on Wed Sep 23 16:53:53 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NICE documentation
==================

NICE is a set of tools designed for calculation of 
invariant and covariant atomic structure representations. It allows to
automatically select most informative combinations of high order spectrum elements
and perform their efficient computation usring recurrency relations. 

Though being designed specifically for atomistic machine learning, NICE in principle 
can be applied to other machine learning tasks which involves signals on sphere/ on a ball
with necessasity to produce invariant/covariant output. 


Theory in a nutshell
____________________

One can use this toolbox as a black box which calculates proper atomic
structure representations. In this case we refer reader to the :ref:`tutorial-label` along with examples folder to borrow 
appropriate hyperparameters for the real life scenarios. 

In order to meaningfully select hypers or desing your calculations some understanding of 
what is going on is required. The most comprehensive description is given in [Ref]_, which
though might appear to be quite time consuming for people not from the field. Thus, this 
section is designed to give short overview of the method without any prooves and unnecesarry
details.
,
For various purposes in atomistic machine learning, there is need to describe atomic environments
by invariant or covariant values. Atomic environment is described by unordered set of 
relative positions of neighbors within given cut-off radius along with their species 
:math:`\{\{\vec{r_1}, \alpha_1\}, \{\vec{r_2}, \alpha_{2}\}... \{\vec{r_n}, \alpha_{n}\}\}`.
The number of neighbors potentially can be varying. The goal is to provide description
of the fixed size consisting of invariant or covariant features with respect
to permutations of atoms of the same specie along rotations of the environment. 

The invariance with respect to permutation of atoms is achieved by introduction of "neighbor 
density functions": 
:math:`\rho_{\alpha}(\vec{r}) = \sum\limits_i g(\vec{r} - \vec{r_i}) \delta_{\alpha, \alpha_i}`,
where :math:`g` is some local function, such as gaussian, or even delta function. After that 
fingerprints are expressed as the functionals of :math:`\rho`.

To deal with neighbor density functions spherical expansion coefficients are introduced:

.. math::
   < \{n, \alpha\} l m | \rho^1> =  \int d\vec{r} R_{n}(\vec{r}) Y_l^m(\hat{r}) \rho_{\alpha}(\vec{r})
, where :math:`\hat{r}` is the unit direction vector, :math:`r = |\vec{r}|`, :math:`R_{n}(r)` is 
some complete basis, not really matters which one particularly, 
:math:`Y_l^m(\hat{r})` are spherical harmonics [#f1]_. :math:`l` index runs from :math:`0` 
to :math:`+\inf`, 
:math:`m` runs from :math:`-l` to :math:`l`.

:math:`\{n, \alpha\}` indices never used separately from each other and, thus, for simplicity, 
in further narrative we will refer to them as to just :math:`n`. 

It is known how coefficients :math:`< n l m | \rho^1>` transforms under rotations of the environment.
Particulary coefficients with :math:`l = 0` remains constants under rotations, i. e. are invariants,
while the general transformation rule is

.. math::
   < n l m | \hat{R} | \rho^1> = \sum\limits_{m'} D^l_{mm'} < n l m' | \rho^1>

where :math:`< n l m | \hat{R} | \rho^1>` are spherical expansion coefficients
for the rotated environment, :math:`\hat{R}'` is the rotation, described, for instance,
by Euler angles  [#f2]_, :math:`D^l_{mm'}(\hat{R})` are Wigner D matrices  [#f3]_. 

Let's look at this transformation more closely. First of all we see that spherical expansion
coefficients of rotated environment depends only on coefficients of the initial environments
with the same :math:`n` and :math:`l` indices. I. e. one can group coefficients into vectors 
corresponding to fixed :math:`n` and :math:`l` of size :math:`2l + 1` and indexed by :math:`m`
index. The transformation itself is nothing else but matrix vector multiplication. 

Within this framework we work only with this way of transformation. Further we will call 
any vector of odd size which transforms this way as covariant feature/fingerprint. 



Some transformations upon covariant vectors leads to also covariant vectors, some not. 
For instance we can apply elementwise squaring of vector elements which clearly would 
result in non covariant vector. 

There are several ways to combine covariants to get covariant output. The most obvious is to
construct linear combination of covariants. 

.. math::
   {output}^l_m = \sum\limits_i (input_i)^l_m * q_i
   

where :math:`q_i` are arbitrarily coefficients. The less obvious way is to do Clebsch-Gordan 
iteration: 

.. math::
   {output}^l_m  = \sum\limits_{m_1 m_2} <l_1 m_1; l_2 m_2| l m> (first\:input)^l_m (second\:input)^l_m

, there :math:`<l_1 m_1; l_2 m_2| l m>` are Clebsch-Gordan coefficients. 

For further purposes it is necessary to introduce the concept of body order.

It is clear that combining transformation rules [] and [] we get covariants
which depends polynomially on the entries of initial spherical expansion coefficients.

if all monomials have the same power :math:`v` than the define body order of the 
corresponding covariant vector to be :math:`v`. If monomials have different powers 
than body order is undefined. 

If we apply linear combination to the covariants of body order :math:`v` than result is also
of body order :math:`v`. If we do Clebsch-Gordan iteration with covariants of body order 
:math:`v_1` and :math:`v_2` than the result has body order :math:`v_1 + v_2`. 

There are several important statements:

1. Completeness a. For any :math:`v` Using combination rule [2] one can get
 the full basis in the 





.. _tutorial-label:

tutorial
________


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. rubric:: Footnotes

.. [#f1] https://en.wikipedia.org/wiki/Spherical_harmonics

.. [#f2] https://en.wikipedia.org/wiki/Euler_angles

.. [#f3] https://en.wikipedia.org/wiki/Wigner_D-matrix


.. [Ref] https://aip.scitation.org/doi/10.1063/5.0021116