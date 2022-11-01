Theory in a nutshell
_____________________________

One can use this toolbox as a black box that calculates proper atomic
structure representations. In this case, we refer the reader to tutorials along with an examples folder to borrow 
appropriate hyperparameters for real-life scenarios. 

In order to meaningfully select hypers or design your calculations, some understanding of 
what is going on is required. The most comprehensive description is given in [Ref]_, which 
might appear to be quite time-consuming for people not from the field. Thus, this 
section is designed to give a short overview of the method without any proofs or unnecessary 
details.

For various purposes in atomistic machine learning, there is a need to describe atomic environments
by invariant or covariant values. The most widespread case is the construction of so-called
machine learning potentials. In this case, the goal is to construct a mapping function from an atomistic structure,
whether it is a molecule, crystal, or amorphous solid to the energy of this configuration. Energy is 
an `extensive <https://en.wikipedia.org/wiki/Intensive_and_extensive_properties>`_ property, which allows representing total energy as a sum of atomic contributions which are defined by central atomic specie along with
the atomic environment. 

Most machine learning algorithms don't exhibit required symmetries such as rotational symmetry out of the box. 
Thus, there is a need to calculate atomic environment representation which is invariant with respect to certain transformations. 
For the prediction of other properties, there is also a need for covariant representations which transform in a certain way under rotations.


The atomic environment is described by an unordered set of 
relative positions of neighbors within a given cut-off radius, along with their species 
:math:`\{\{\vec{r_1}, \alpha_1\}, \{\vec{r_2}, \alpha_{2}\}... \{\vec{r_n}, \alpha_{n}\}\}`.
The number of neighbors potentially can vary. The goal is to provide a description
of the fixed size consisting of invariant or covariant features with respect
to permutations of atoms of the same specie along rotations of the environment. 

The invariance with respect to the permutation of atoms is achieved by introducing "neighbor 
density functions": 
:math:`\rho_{\alpha}(\vec{r}) = \sum\limits_i g(\vec{r} - \vec{r_i}) \delta_{\alpha, \alpha_i}`,
where :math:`g` is some local function, such as a gaussian or even delta function. After that 
fingerprints are expressed as the functionals of :math:`\rho`.

To deal with neighbor density functions, spherical expansion coefficients are introduced:

.. math::
   < \{n, \alpha\} \lambda m | \rho^1> =  \int d\vec{r} R_{n}(\vec{r}) Y_{\lambda}^m(\hat{r}) \rho_{\alpha}(\vec{r})
, where :math:`\hat{r}` is the unit direction vector, :math:`r = |\vec{r}|`, :math:`R_{n}(r)` is 
some complete basis, it doesn't really matter which one particularly, 
:math:`Y_{\lambda}^m(\hat{r})` are
`spherical harmonics <https://en.wikipedia.org/wiki/Spherical_harmonics>`_.  :math:`\lambda` index runs from :math:`0` 
to :math:`+\inf`, 
:math:`m` runs from :math:`-\lambda` to :math:`\lambda`.

:math:`\{n, \alpha\}` indices are never used separately from each other and, thus, for simplicity, 
in the further narrative, we will refer to them as just :math:`n`. 

It is known how coefficients :math:`< n \lambda m | \rho^1>` transform under rotations of the environment.
Particularly coefficients with :math:`l = 0` remain constant under rotations, i. e. are invariants,
while the general transformation rule is

.. math::
   < n \lambda m | \hat{R} | \rho^1> = \sum\limits_{m'} D^{\lambda}_{mm'} < n \lambda m' | \rho^1>

where :math:`< n \lambda m | \hat{R} | \rho^1>` are spherical expansion coefficients
for the rotated environment, :math:`\hat{R}` is the rotation, described, for instance,
by `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_
, :math:`D^{\lambda}_{mm'}(\hat{R})` are
`Wigner D matrices <https://en.wikipedia.org/wiki/Wigner_D-matrix>`_. 

Let's look at this transformation more closely. First of all, we see that spherical expansion
coefficients of the rotated environment depend only on coefficients of the initial environments
with the same :math:`n` and :math:`\lambda` indices. I. e., one can group coefficients into vectors 
corresponding to fixed :math:`n` and :math:`\lambda` of size :math:`2 \lambda + 1` and indexed by :math: 'm'
index. The transformation itself is nothing else but matrix-vector multiplication. 

Within this framework, we work only with this way of transformation. Further, we will call 
any vector of odd size which transforms this way as a covariant feature/fingerprint. 



Some transformations upon covariant vectors also lead to covariant vectors. Some do not. 
For instance, we can apply elementwise squaring of vector elements which clearly would 
result in a non-covariant vector. 

There are several ways to combine covariants to get a covariant output. The most obvious is to
construct a linear combination of covariants. 

.. math:: 
   :label: first_expansion

   {output}^{\lambda}_m = \sum\limits_i (input_i)^{\lambda}_m * q_i

   
   

where :math:`q_i` are arbitrarily coefficients. The less obvious way is to do a Clebsch-Gordan 
iteration: 

.. math::
   :label: second_expansion

   {output}^{\lambda}_m  = \sum\limits_{m_1 m_2} <l_1 m_1; l_2 m_2| \lambda m>
    (first\:input)^{l_1}_{m_1} (second\:input)^{l_2}_{m_2}

, there :math:`<l_1 m_1; l_2 m_2| \lambda m>` are
`Clebsch-Gordan coefficients <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`_. 

Let's take a look at the second construction rule in more detail. It takes 
two covariant vectors as input and constructs several covariant outputs, indexed
by natural index :math:`\lambda`. (Actually, :math:`\lambda` is bounded between 
:math:`| l_1 - l_2 |` and :math:`|l_1 + l_2|`, otherwise Clebsch-Gordan coefficients are zeros)


For further purposes, it is necessary to introduce the concept of body order.

It is clear that by combining transformation rules :eq:`first_expansion` and :eq:`second_expansion`, we get covariants
which depend polynomially on the entries of initial spherical expansion coefficients.

If all monomials have the same power :math:`\nu`, then we define the body order of the 
corresponding covariant vector to be :math:`\nu`. If monomials have different powers, 
then body order is undefined. 

If we apply the linear combination to the covariants of body order :math:`\nu`, then the result also 
has a body order :math:`\nu`. If we do Clebsch-Gordan iteration with covariants of body order 
:math:`\nu_1` and :math:`\nu_2`, then the result has body order :math:`\nu_1 + \nu_2`. 

Consider the following procedure. Initially, we 
have :math:`\nu = 1`, and initial spherical expansion
coefficients :math:`< n \lambda m | \rho^1>` . Let's apply the construction rule
:eq:`second_expansion` for each pair of spherical expansion coefficients
and for each possible output :math:`\lambda`. The result would be set 
of :math:`\nu=2` body order covariants. As the next step, let's do the same 
for each pair of the obtained :math:`\nu=2` covariants and
initial :math:`\nu=1` spherical expansion coefficients. The result would
be a set of :math:`\nu=3` covariants. And so on. 


There are two important statements:

1. Completeness a. 
   For each :math:`\nu` set of covariants obtained by the previously discussed procedure is complete basis in the space of :math:`v` order functionals 
   from :math:`\rho(*)` to invariant/covariant output. It means
   that any :math:`\nu` order functional can be expressed as a linear combination 
   of  :math:`\nu` order covariants/invariants. 

2. Completeness b. 
   For each :math:`\nu` set of covariants obtained by the previously discussed 
   procedure is a complete basis in a space of :math:`v` body order potentials.
   It means that any function of atomic structure given by the sum of contributions
   over all subsets of :math:`\nu` atoms can be represented as the linear
   combination of :math:`\nu` order covariants/invariants. Particularly any 
   two-body potential, such as `LJ potential <https://en.wikipedia.org/wiki/Lennard-Jones_potential>`_,
   can be represented as 
   linear combination of first-order invariants, any three-body potential 
   can be represented as a linear combination of second-order invariants
   and so on.


Taking into account these facts, it looks like the recipe for machine learning
potentials is very clear. Just iterate over the body order
until convergence. 

The problem is that the size of :math:`\nu` order covariants explodes with 
:math:`\nu` exponentially. Indeed, when we go from :math:`\nu - 1` to
:math:`\nu` order number of entries is multiplied by the number 
of :math:`\nu=1` order covariant vectors and by the number of 
different :math:`\lambda`-s. Thus, it is not computationally feasible to
go to high body orders with this naive approach.

In practice, for particular distributions in phase space, given by particular
datasets, by far, not all components of covariants are relevant. Namely,
in real-life scenarios the `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
spectrum decreases very rapidly. So, 
in fact, we need only a few components out of a great many. 

There is a way to construct iterative components iteratively. 
It consists of iterative PCA and Clebsch-Gordan expansions. For each 
transition from :math:`\nu-1` body order to :math:`\nu` body order, we do PCA 
of :math:`\nu-1` body order covariants and use only those with the highest
variance or importance for subsequent expansion. The number of components
to take can be either fixed or selected dynamically in such a way as to cover a certain percentage of the 
variance in the dataset. 

It is clear that in this way, most part of the variance is kept. Indeed,
let's imagine that we had exact linear dependencies at some step, and, thus,
after PCA, some components have exact zero variance. Substituting vector with zeros to the 
expansion rule :eq:`second_expansion` we see that the result is ... also zeros. 
The same relates to small components - components with small variance also 
"give birth" to components with small variance. Thus, neglecting them 
would not affect the covariants with higher body orders much. 

There is another important observation that on a particular dataset, covariants with different body orders can correlate with each other. Thus,
it is a good idea to preserve at each iteration not the components with
the highest absolute variance but the components with the
highest "purified variance" or "new variance". I. e. components 
with the highest residuals, which can not be explained by linear regression
based on previous body orders. Using 
"`sklearn <https://scikit-learn.org/stable/>`_ language" purification
step can be viewed as :

.. code-block:: python

   purified_covariants = covariants - linear_regressor.fit(
       all_covariants_of_smaller_body_order, covariants).predict(covariants)


To conclude, NICE consist of iterations each of three steps:

1. Expansion - raising the body order by one using Clebsh-Gordan iteration :eq:`second_expansion`.
2. Purification - getting rid of variance, which is explainable by previous body-order covariants.
3. PCA - to group the most part of the variance in a small subset of components.


In principle, one can apply this machinery to other invariant/covariant machine learning tasks
not related to atomistic machine learning. The only difference is that in this case, 
input spherical expansion coefficients :math:`< n \lambda m | \rho^1>` would be obtained from 
some other sphere/ball signal, not from the sum of Gaussians as in the case of atomistic machine learning. 

In the current implementation there is also a duplicate branch of only invariants, 
which allows choosing hyper parameters, such as the number of components to expand,
separately for invariants and covariants, which is very useful in practice. 

More about it in the first tutorial, "Constructing machine learning potential".



.. [Ref] https://aip.scitation.org/doi/10.1063/5.0021116
