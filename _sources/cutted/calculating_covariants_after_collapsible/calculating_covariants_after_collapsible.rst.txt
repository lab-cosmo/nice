.. code:: ipython3

    data_even, data_odd, invariants_even = transformers[1].transform(train_coefficients[1])

.. code:: ipython3

    print(data_even.covariants_.shape)
    print(data_even.actual_sizes_)
    
    
    print(data_odd.covariants_.shape)
    print(data_odd.actual_sizes_)
    
    for key in invariants_even.keys():
        print(invariants_even[key].shape)


.. parsed-literal::

    (40000, 87, 6, 11)
    [22 55 73 83 87 76]
    (40000, 88, 6, 11)
    [20 54 72 87 88 75]
    (40000, 10)
    (40000, 200)
    (40000, 200)
    (40000, 200)


.. code:: ipython3

    data_true, data_pseudo = ParityDefinitionChanger().transform(data_even, data_odd)
    
    print(data_true.covariants_.shape)
    print(data_true.actual_sizes_)
    
    print(data_pseudo.covariants_.shape)
    print(data_pseudo.actual_sizes_)


.. parsed-literal::

    (40000, 87, 6, 11)
    [22 54 73 87 87 75]
    (40000, 88, 6, 11)
    [20 55 72 83 88 76]


.. code:: ipython3

    data_even, data_odd = ParityDefinitionChanger().transform(data_true, data_pseudo)

.. code:: ipython3

    for lambd in range(6):
        data_true.covariants_[:, :data_true.actual_sizes_[lambd],
                              lambd, :(2 * lambd + 1)] /= (2 * lambd + 1)
        data_pseudo.covariants_[:, :data_pseudo.actual_sizes_[lambd],
                                lambd, :(2 * lambd + 1)] /= (2 * lambd + 1)
