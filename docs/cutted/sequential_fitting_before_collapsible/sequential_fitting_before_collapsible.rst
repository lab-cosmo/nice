Sequential fitting
------------------

It is not always clear how to select good hyperparameters for
calculations. The second tutorial “Getting insights about the model,”
showed how to plot PCA spectrums for all lambda channels and parities.
This information, along with the other one, such as regression accuracy,
might be useful to select better hypers. Particularly, the most
straightforward way is to select the number of PCA components in such a
way as to cover the most part of the variance and do it successively
from block to block.

In this case, it is very undesirable to fit all parts of the model,
including not changed ones from scratch. One possible way around is to
do all things by hand, as was described in the tutorial “Constructor or
non standard_sequence,” but there would be an additional headache with
packing resulting blocks into a single model with a convenient
.transform method. Nice toolbox has the capability to do it very
succinctly.

First of all, we need to get spherical expansion coefficients the same
way as in previous tutorials:
