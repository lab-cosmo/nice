Calculating covariants
----------------------

In the previous tutorial, we calculated invariant representations of
atomic environments and used them for the prediction of energies -
invariant properties.

In the case when there is a need to predict covariant properties,
covariants instead of invariants are required. This tutorial shows how
to calculate them.

First of all, we need to get **fitted** instance of the model as in the
previous tutorial. It is done by the following preliminaries cell: (with
the only difference that since we want to calculate covariants, we
clearly shouldn’t leave the covariants branch of the last block empty)
