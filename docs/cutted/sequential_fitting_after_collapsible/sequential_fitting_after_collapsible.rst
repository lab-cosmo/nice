coefficients are now spherical expansion coefficients for H centered
environments:

.. code:: ipython3

    print(coefficients.shape)


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        (4000, 10, 6, 11)
    </p>
    </pre>
    </embed>
    
Let’s do the first steps from standar sequence:

.. code:: ipython3

    even_0, odd_0 = InitialTransformer().transform(coefficients)
    initial_pca = IndividualLambdaPCAsBoth()
    initial_pca.fit(even_0, odd_0)
    even_0_t, odd_0_t = initial_pca.transform(even_0, odd_0)

Now we can fit couple of standard blocks:

.. code:: ipython3

    block_1 = StandardBlock(ThresholdExpansioner(100), None,
                            IndividualLambdaPCAsBoth(20))
    block_1.fit(even_0_t, odd_0_t, even_0_t, odd_0_t)
    even_1, odd_1, _ = block_1.transform(even_0_t, odd_0_t, even_0_t, odd_0_t)
    
    block_2 = StandardBlock(None, None, None,
                            ThresholdExpansioner(100, mode='invariants'))
    block_2.fit(even_1, odd_1, even_0_t, odd_0_t)
    _, _, even_invariants = block_2.transform(even_1, odd_1, even_0_t, odd_0_t)

At his moment we have all parts of this standard sequence fitted:

.. code:: ipython3

    nice = StandardSequence(initial_pca=initial_pca, blocks=[block_1, block_2])
    print(initial_pca.is_fitted())
    print(block_1.is_fitted())
    print(block_2.is_fitted())


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        True
        True
        True
    </p>
    </pre>
    </embed>
    
what about full model?

.. code:: ipython3

    print(nice.is_fitted())


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        False
    </p>
    </pre>
    </embed>
    
Nope.

At this point, there is a very high probability of making a mistake.
Particularly one can feed StandardSequence with some fitted initial_pca
along with blocks, which were fitted based not on the same initial_pca,
with different initial_normalizer, or even on different data. In order
to prevent it, there is a requirement to pass an additional flag
guaranteed_parts_fitted_consistently = True to the model:

.. code:: ipython3

    nice = StandardSequence(initial_pca=initial_pca,
                            blocks=[block_1, block_2],
                            guaranteed_parts_fitted_consistently=True)
    print(nice.is_fitted())


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        True
    </p>
    </pre>
    </embed>
    
Model is considered to be fitted if 1) all parts are fitted and 2) if
guaranteed_parts_fitted_consistently is set to be True

**Golden rule:** Every time you pass
guaranteed_parts_fitted_consistently = True make a pause and think
twice.

Let’s check consistency:

.. code:: ipython3

    even_invariants_2 = nice.transform(coefficients,
                                       return_only_invariants=True)[3]
    print(np.sum(np.abs(even_invariants - even_invariants_2)))


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        0.0
    </p>
    </pre>
    </embed>
    
This also works in other direction:

.. code:: ipython3

    initial_pca = IndividualLambdaPCAsBoth()
    block_1 = StandardBlock(ThresholdExpansioner(100), None,
                            IndividualLambdaPCAsBoth(20))
    block_2 = StandardBlock(None, None, None,
                            ThresholdExpansioner(100, mode='invariants'))
    
    print(initial_pca.is_fitted())
    print(block_1.is_fitted())
    print(block_2.is_fitted())
    
    nice = StandardSequence(initial_pca=initial_pca, blocks=[block_1, block_2])
    nice.fit(coefficients)
    
    print(initial_pca.is_fitted())
    print(block_1.is_fitted())
    print(block_2.is_fitted())


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        False
        False
        False
        True
        True
        True
    </p>
    </pre>
    </embed>
    
StandardBlock behaves the same way:

.. code:: ipython3

    expansioner, pca = ThresholdExpansioner(100), IndividualLambdaPCAsBoth(20)
    print(expansioner.is_fitted())
    print(pca.is_fitted())
    
    block = StandardBlock(expansioner, None, pca)
    block.fit(even_0_t, odd_0_t, even_0_t, odd_0_t)
    
    print(expansioner.is_fitted())
    print(pca.is_fitted())


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        False
        False
        True
        True
    </p>
    </pre>
    </embed>
    
.. code:: ipython3

    expansioner, pca = ThresholdExpansioner(100), IndividualLambdaPCAsBoth(20)
    expansioner.fit(even_0_t, odd_0_t, even_0_t, odd_0_t)
    even_1, odd_1 = expansioner.transform(even_0_t, odd_0_t, even_0_t, odd_0_t)
    pca.fit(even_1, odd_1)
    
    block = StandardBlock(expansioner,
                          None,
                          pca,
                          guaranteed_parts_fitted_consistently=True)
    
    print(block.is_fitted())


.. raw:: html
    
    <embed>
    <pre>
    <p style="margin-left: 5%;font-size:12px;line-height: 1.2; overflow:auto" >
        True
    </p>
    </pre>
    </embed>
    
There is another group of blocks that accepts classes, such as
sklearn.linear_model.Ridge in the initialization. But in their case,
there is a need to apply several distinct regressors separately for each
lambda channel and parity. Thus, the input regressor is cloned, and
initial instances are not touched in any way. So, the material of this
tutorial does not apply to purifiers.

