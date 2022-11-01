Custom regressors into purifiers
--------------------------------

As was already mentioned in the first tutorial, purifiers can accept
arbitrarily linear regressors from sklearn.linear_model. In order to
feed it with a custom linear regressor, some requirements should be
fulfilled. Firstly, it should have the same interface as linear
regressors from sklearn with the fit and predict methods. Secondly, it
should fulfill sklearn requirements to make it possible to clone with
`sklearn.base.clone <https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html>`__
function. This tutorial shows an example of such a class.

As before, let’s calculate spherical expansion coefficients for H
environments:
