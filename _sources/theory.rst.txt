Theory in a nutshell
_____________________________

One can use this toolbox as a black box which calculates proper atomic
structure representations. In this case we refer reader to tutorials along with examples folder to borrow 
appropriate hyperparameters for the real life scenarios. 

In order to meaningfully select hypers or desing your calculations some understanding of 
what is going on is required. The most comprehensive description is given in [Ref]_, which
though might appear to be quite time consuming for people not from the field. Thus, this 
section is designed to give short overview of the method without any proves and unnecesarry
details.

For various purposes in atomistic machine learning, there is need to describe atomic environments
by invariant or covariant values. The most widespread case is construction of so-called
machine learning potentials. In this case goal is to construct mapping function from atomistic structure,
whether it is molecule, crystal or amorphous solid to energy of this configuration. Energy is 
`extensive <https://en.wikipedia.org/wiki/Intensive_and_extensive_properties>`_ property, which allows to 
represent total energy as a sum of atomic contributions which are defined by central atomic specie along with
atomic environment. 

Most of machine learning algorithms doesn't exhibit required symmetries such as rotational symmetry out of the box. 
Thus, there is need to calculate atomic environment representation which is invariant with respect to certain transformations. 
For prediction of other properties there is also need in covariant representations which transforms in certain way under rotations.


Atomic environment is described by unordered set of 
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
:math:`Y_l^m(\hat{r})` are
`spherical harmonics <https://en.wikipedia.org/wiki/Spherical_harmonics>`_.  :math:`l` index runs from :math:`0` 
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
for the rotated environment, :math:`\hat{R}` is the rotation, described, for instance,
by `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_
, :math:`D^l_{mm'}(\hat{R})` are
`Wigner D matrices <https://en.wikipedia.org/wiki/Wigner_D-matrix>`_. 

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
   :label: first_expansion

   {output}^l_m = \sum\limits_i (input_i)^l_m * q_i

   
   

where :math:`q_i` are arbitrarily coefficients. The less obvious way is to do Clebsch-Gordan 
iteration: 

.. math::
   :label: second_expansion

   {output}^{\lambda}_m  = \sum\limits_{m_1 m_2} <l_1 m_1; l_2 m_2| \lambda m>
    (first\:input)^{l_1}_{m_1} (second\:input)^{l_2}_{m_2}

, there :math:`<l_1 m_1; l_2 m_2| l m>` are
`Clebsch-Gordan coefficients <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`_. 

Let's take a loot at the second construction rule in more detail. It takes as 
the input two covariant vectors and constructs several covariant outputs, indexed
by natural index :math:`\lambda`. (Actually, :math:`\lambda` is bounded between 
:math:`| l_1 - l_2 |` and :math:`|l_1 + l_2|`, otherwise Clebsch-Gordan coefficients are zeros)


For further purposes it is necessary to introduce the concept of body order.

It is clear that combining transformation rules :eq:`first_expansion` and :eq:`second_expansion` we get covariants
which depends polynomially on the entries of initial spherical expansion coefficients.

if all monomials have the same power :math:`\nu` than the define body order of the 
corresponding covariant vector to be :math:`\nu`. If monomials have different powers 
than body order is undefined. 

If we apply linear combination to the covariants of body order :math:`\nu` than result is also
of body order :math:`\nu`. If we do Clebsch-Gordan iteration with covariants of body order 
:math:`\nu_1` and :math:`\nu_2` than the result has body order :math:`\nu_1 + \nu_2`. 

Consider the following procedure. Initially we 
have :math:`\nu = 1`, and initial spherical expansion
coefficients :math:`< n l m | \rho^1>` . Let's apply construction rule
:eq:`second_expansion` for each pair of spherical expansion coefficients,
and for each possible :math:`\lambda`. The result would be set 
of :math:`\nu=2` body order covariants. As the next step let's do the same 
for each pair of the obtained :math:`\nu=2` covariants, and
initial :math:`\nu=1` spherical expansion coefficients. The result would
be set of :math:`\nu=3` covariants. And so on. 


There are two important statements:

1. Completeness a. 
   For each :math:`\nu` set of covariants obtained by previously discussed 
   procedure is complete basis in the space of :math:`v` order functinonals
   from :math:`\rho(*)` to invariant/covariant output. It means
   that any :math:`\nu` order functional can be expressed as linear combination 
   of  :math:`\nu` order covariants/invariants. 

2. Completeness b. 
   For each :math:`\nu` set of covariants obtained by previously discussed 
   procedure is complete basis in a space of :math:`v` body order potentials.
   It means, that any function of atomic structure given by sum of contributions
   over all subsets of :math:`\nu` atoms can be represented as the linear
   combination of :math:`\nu` order covariants/invariants. Particularly any 
   two-body potential, such as `LJ potential <https://en.wikipedia.org/wiki/Lennard-Jones_potential>`_,
   can be represented as 
   linear combination of first order invariants, any three-body potential 
   can be represented as linear combination of second order invariants
   and so on.


Taking into account these facts, it looks like that the recipe for machine learning
potentials is very clear. Just iterate over the body order
until convergence. 

The problem is that the size of :math:`\nu` order covariants explodes with 
:math:`\nu` exponentially. Indeed, when we go from :math:`\nu - 1` to
:math:`\nu` order number of entries is multiplied by the number 
of :math:`\nu=1` order covariant vectors and by the number of 
different :math:`\lambda`-s. Thus, it is not computationaly feasible to
go to high body orders with naive approach.

In practice, for particular distributions in phase space, given by particular
datasets, by far not all components of covariants are relevant. Namely,
in real life scenarious the `PCA <https://en.wikipedia.org/wiki/Principal_component_analysis>`_
spectrum decreases very rapidly. So, 
in fact, we need only few components out of great many. 

There is a way how to construct iterative components iteratively. 
It consist of iterative PCA and Clebsch-Gordan expansions. For each 
transition from :math:`\nu-1` body order to :math:`\nu` body oder we do PCA 
of :math:`\nu-1` body order covariants and use only ones with the highest
variance/importance for subsequent expansion. The number of components
to take can be either fixed either to cover certain percentabe of the 
variance in the dataset. 

It is clear that this way the most part of variance is keeped. Indeed,
let's imagine that we had exact linear dependencies at some step, and, thus,
after pca some components have exact zero variance. Substituting zero to 
expansion rule :eq:`second_expansion` we see that result is ... also zeros. 
The same relates to small components - components with small variance 
"give a birth" to components with also small variance, thus their neglecting,
would not affect much covariants with higher body orders. 

There is one another important observation, that on particular dataset's
covariants with different body orders can correlate with each other. Thus,
it is a good idea, to preserve at each iteration, not the components with
the highest absolute variance, but the components with the
highest "purified variance" or "new variance". I. e. components 
with highest residuals, which can not be explained by linear regression
based on previous body orders. Using 
"`sklearn <https://scikit-learn.org/stable/>`_ language" purification
step can be viewed as :

.. code-block:: python

   purified_covariants = covariants - linear_regressor.fit(
       all_covariants_of_smaller_body_order, covariants).predict(covariants)


To conclude, NICE consist of iterations each of three steps:

1. Expansion - raising the body order by one using Clebsh-Gordan iteration :eq:`second_expansion`.
2. Purification - getting rid of variance, which is explainable by previous body-order covariants.
3. PCA - to group the most part of variance in small subset of components.


In principle one can apply this machinery to other invariant/covariant machine learning tasks
not related to atomistic machine learning.  The only difference is that in this case 
input spherical expansion coefficients :math:`< n l m | \rho^1>` would be obtained from 
some other sphere/ball signal, not from sum of gaussians as in case of atomistic machine learning. 

In current implementation there is also duplicate branch of only invariants, 
which allows to choose hyper parameters, such as the amount of components to expand,
separatelly for invariants and covariants, which is very usefull in practice. 

More about it in the first tutorial "Constructing machine learning potential".



.. [Ref] https://aip.scitation.org/doi/10.1063/5.0021116
