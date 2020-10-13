coefficients are now spherical expansion coefficients for H centered
environments:

.. code:: ipython3

    print(coefficients.shape)


.. parsed-literal::

    (4000, 10, 6, 11)


Let's do the first steps from standar sequence:

.. code:: ipython3

    even_0, odd_0 = InitialTransformer().transform(coefficients)
    initial_pca = IndividualLambdaPCAsBoth()
    initial_pca.fit(even_0, odd_0)
    even_0_t, odd_0_t = initial_pca.transform(even_0, odd_0)

Now we can fit couple of standard blocks:

.. code:: ipython3

    block_1 = StandardBlock(ThresholdExpansioner(100), None, IndividualLambdaPCAsBoth(20))
    block_1.fit(even_0_t, odd_0_t, even_0_t, odd_0_t)
    even_1, odd_1, _ = block_1.transform(even_0_t, odd_0_t, even_0_t, odd_0_t)
    
    block_2 = StandardBlock(None, None, None, ThresholdExpansioner(100, mode = 'invariants'))
    block_2.fit(even_1, odd_1, even_0_t, odd_0_t)
    _, _, even_invariants = block_2.transform(even_1, odd_1, even_0_t, odd_0_t)

At his moment we have all parts of this standard sequence fitted:

.. code:: ipython3

    trans = StandardSequence(initial_pca= initial_pca, blocks = [block_1, block_2])
    print(initial_pca.is_fitted())
    print(block_1.is_fitted())
    print(block_2.is_fitted())


.. parsed-literal::

    True
    True
    True


what about full model?

.. code:: ipython3

    print(trans.is_fitted())


.. parsed-literal::

    False


Nope.

At this point there is very high probability to make a mistake.
Particularly one can feed StandardSequence with some fitted initial\_pca
along with blocks, which were fitted based not on the same initial\_pca,
or with different initial\_normalizer, or even on different data. In
order to prevent it, there is requirement to pass additional flag
guaranteed\_parts\_fitted\_consistently = True to the model:

.. code:: ipython3

    trans = StandardSequence(initial_pca= initial_pca, blocks = [block_1, block_2],
                             guaranteed_parts_fitted_consistently = True)
    print(trans.is_fitted())


.. parsed-literal::

    True


Model is considered to be fitted if 1) all parts are fitted and 2) if
guaranteed\_parts\_fitted\_consistently is set to be True

**Golden rule:** Every time you pass
guaranteed\_parts\_fitted\_consistently = True make a pause and think
twice.

Let's check consistency:

.. code:: ipython3

    even_invariants_2 = trans.transform(coefficients, return_only_invariants = True)[3]
    print(np.sum(np.abs(even_invariants - even_invariants_2)))


.. parsed-literal::

    0.0


This also works in other direction:

.. code:: ipython3

    initial_pca = IndividualLambdaPCAsBoth()
    block_1 = StandardBlock(ThresholdExpansioner(100), None, IndividualLambdaPCAsBoth(20))
    block_2 = StandardBlock(None, None, None, ThresholdExpansioner(100, mode = 'invariants'))
    
    print(initial_pca.is_fitted())
    print(block_1.is_fitted())
    print(block_2.is_fitted())
    
    trans = StandardSequence(initial_pca = initial_pca, blocks = [block_1, block_2])
    trans.fit(coefficients)
    
    print(initial_pca.is_fitted())
    print(block_1.is_fitted())
    print(block_2.is_fitted())


.. parsed-literal::

    False
    False
    False
    True
    True
    True


StandardBlock behaves the same way:

.. code:: ipython3

    expansioner, pca = ThresholdExpansioner(100), IndividualLambdaPCAsBoth(20)
    print(expansioner.is_fitted())
    print(pca.is_fitted())
    
    block = StandardBlock(expansioner, None, pca)
    block.fit(even_0_t, odd_0_t, even_0_t, odd_0_t)
    
    print(expansioner.is_fitted())
    print(pca.is_fitted())


.. parsed-literal::

    False
    False
    True
    True


.. code:: ipython3

    expansioner, pca = ThresholdExpansioner(100), IndividualLambdaPCAsBoth(20)
    expansioner.fit(even_0_t, odd_0_t, even_0_t, odd_0_t)
    even_1, odd_1 = expansioner.transform(even_0_t, odd_0_t, even_0_t, odd_0_t)
    pca.fit(even_1, odd_1)
    
    block = StandardBlock(expansioner, None, pca, 
                          guaranteed_parts_fitted_consistently = True)
    
    print(block.is_fitted())


.. parsed-literal::

    True


There is another group of blocks, which accepts classes such as
sklearn.linear\_model.Ridge in the initialization. But in their case
there is need to apply several distinct regressors, separatelly for each
lambda channel and parity. Thus, input regressor is clonned, and initial
instances is not touched in any way. So, the material of this tutorials
does not apply to purifiers.
