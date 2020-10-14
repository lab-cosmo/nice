Custom regressors into purifiers
--------------------------------

As it was already mentioned in the first tutorial, purifers can accept
arbitrarily linear regressor form sklearn.linear\_model. In order to
feed it with custom linear regressor some requirements should be
fulfilled. Firstly, it should have the same interface as linear
regressors from sklearn with fit and predict methods. Secondly, it
should fullfill sklearn requiremenets to make it possible to clone with
`sklearn.base.clone <https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html>`__
function. This tutorial shows an example of such class.

As before let's calculate spherical expansion coefficients for H
environments:
