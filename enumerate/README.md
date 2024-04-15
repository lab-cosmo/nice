Enumeration of linearly and polynomially-independent invariants
===============================================================

The Mathematica notebooks  `linear_reduce.nb` and `polynomial_reduce.nb` 
use computer algebra to list all of the coefficients of density-correlation
equivariants that are linearly independent, or that cannot be computed as
polynomial of lower-order invariants.

The code is not optimized, and cannot go beyond relatively low body order
and nmax,lmax thresholds. It shows however how there are relatively few
invariants that can be dropped beyond those that can be identified based
on angular momentum recoupling theory.

The repository also contains a few examples of the listings, named as
`indep-nmax-lmax.dat`
Entries in each file list the indices of the nonzero invariants, labeled as

```
# nu sigma lambda n1 l1 k1 [n2 l2 k2 .....]
```

following the notation from https://arxiv.org/abs/2007.03407
