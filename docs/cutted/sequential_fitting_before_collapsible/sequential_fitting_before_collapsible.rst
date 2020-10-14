Sequential fitting
------------------

It is not always clear how to select good hyperparameters for
calculations. In the second tutorial "Getting insights about the model"
it was shown how to plot spectrums of PCA for all lambda channels and
parities. This information along with the other one, such as regression
accuracy might be usefull to select better hypers. Particulary, the most
straghtforward way is to select number of pca components, in a such a
way to cover the most part of variance, and do it successively from
block to block.

In this case it is very undesirable to fit all parts of the model,
including not changed ones from scratch. One possible way around is to
do all things by hand, as it was described in the tutorial "Constructor
or non standard\_sequence", but there would be additional headache with
packing resulting blocks into a single model with convenient .transform
method. Nice toolbox has capabilities to do it very succinctly.

First of all we need to get spherical expansion coefficients the same
way as in previous tutorials:
